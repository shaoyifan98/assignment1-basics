from ast import pattern
from collections.abc import Iterable, Iterator
import os
from collections import Counter, defaultdict
from multiprocessing import Pool
import pickle

import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _count_pretokens(text: str, special_tokens: list[str]) -> Counter:
    """Count pre-tokens in text. Special tokens are split out first so that
    no pre-token (and hence no merge) can span across them."""
    counts: Counter = Counter()
    if special_tokens:
        # longest first, so overlapping special tokens match correctly
        pattern = "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
        docs = re.split(pattern, text)
    else:
        docs = [text]

    for doc in docs:
        for m in re.finditer(GPT2_PAT, doc):
            counts[tuple(m.group().encode("utf-8"))] += 1  # tuple of byte values (ints)
    return counts


def _worker(args: tuple[str, int, int, list[str]]) -> Counter:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return _count_pretokens(chunk, special_tokens)


def _merge_word(word: list[int], a: int, b: int, new_id: int) -> list[int]:
    out = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab: dict[int, bytes] = {}
        # Initialize the vocabulary with single-byte tokens and special tokens
        for i in range(256):
            self.vocab[i] = bytes([i])
        for token in special_tokens:
            self.vocab[len(self.vocab)] = token.encode("utf-8")

    def process(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        pre_token_counts = self._pre_tokenize()
        return self._merge(pre_token_counts)

    def _pre_tokenize(self) -> dict[tuple[int, ...], int]:
        num_processes = os.cpu_count() or 4
        split_token = b"<|endoftext|>"

        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, split_token)

        tasks = [
            (self.input_path, start, end, self.special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        total: Counter = Counter()
        with Pool(num_processes) as pool:
            for c in pool.imap_unordered(_worker, tasks):
                total.update(c)
        return dict(total)

    def _merge(
        self, pre_token_counts: dict[tuple[int, ...], int]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Run BPE merges; return (vocab, merges)."""
        words: list[list[int]] = [list(w) for w in pre_token_counts]
        freqs: list[int] = list(pre_token_counts.values())

        # pair -> total count, and pair -> indices of words that (may) contain it
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)
        for idx, (w, f) in enumerate(zip(words, freqs)):
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                pair_to_words[p].add(idx)

        merges: list[tuple[bytes, bytes]] = []
        num_merges = self.vocab_size - len(self.vocab)

        for _ in range(num_merges):
            if not pair_counts:
                break

            # highest count; ties broken by lexicographically greater (bytes, bytes)
            best = max(
                pair_counts,
                key=lambda p: (pair_counts[p], self.vocab[p[0]], self.vocab[p[1]]),
            )
            a, b = best
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            merges.append((self.vocab[a], self.vocab[b]))

            # only touch words that contain the pair
            for idx in pair_to_words.pop(best):
                w, f = words[idx], freqs[idx]

                # remove this word's old pair contributions
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    if pair_counts[p] == 0:
                        del pair_counts[p]

                new_w = _merge_word(w, a, b, new_id)
                words[idx] = new_w

                # add the new pair contributions
                for p in zip(new_w, new_w[1:]):
                    pair_counts[p] += f
                    pair_to_words[p].add(idx)

        return self.vocab, merges



class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.inv_vocab = {t: i for i, t in self.vocab.items()}

        # special token 不在 vocab 里就追加
        for s in self.special_tokens:
            b = s.encode("utf-8")
            if b not in self.inv_vocab:
                new_id = len(self.vocab)
                self.vocab[new_id] = b
                self.inv_vocab[b] = new_id

        # (bytes, bytes) -> 优先级，越小越先合并
        self.rank = {pair: r for r, pair in enumerate(merges)}

        self._special_set = set(self.special_tokens)
        if self.special_tokens:
            pat = "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
            self._special_re = re.compile(f"({pat})")  # 捕获组：保留 special token
        else:
            self._special_re = None

        self._cache: dict[bytes, list[int]] = {}

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, piece: bytes) -> list[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached

        parts = [bytes([b]) for b in piece]          # 从单字节开始
        while len(parts) > 1:
            best_rank, best_pair = None, None
            for i in range(len(parts) - 1):
                r = self.rank.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, (parts[i], parts[i + 1])
            if best_pair is None:
                break
            # 合并这个 pair 的所有出现位置（从左到右，不重叠）
            new, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best_pair:
                    new.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    new.append(parts[i])
                    i += 1
            parts = new

        ids = [self.inv_vocab[p] for p in parts]
        self._cache[piece] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        segments = self._special_re.split(text) if self._special_re else [text]
        ids: list[int] = []
        for seg in segments:
            if not seg:
                continue
            if seg in self._special_set:
                ids.append(self.inv_vocab[seg.encode("utf-8")])
            else:
                for m in re.finditer(GPT2_PAT, seg):
                    ids.extend(self._encode_pretoken(m.group().encode("utf-8")))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")