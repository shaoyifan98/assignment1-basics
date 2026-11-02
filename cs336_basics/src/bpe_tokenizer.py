import os
from collections import Counter, defaultdict
from multiprocessing import Pool

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