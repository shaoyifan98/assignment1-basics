import json
import time
import pickle
import resource  # Linux/macOS

from cs336_basics.src.bpe_tokenizer import BPETokenizer

if __name__ == "__main__":  # macOS/Windows 必须有这行
    t0 = time.time()
    tok = BPETokenizer(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )
    vocab, merges = tok.process()
    elapsed = time.time() - t0

    # ru_maxrss: Linux 单位是 KB，macOS 单位是 bytes
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"time: {elapsed:.1f}s, peak RSS (主进程): {peak}")

    with open("tinystories_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("tinystories_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    longest = max(vocab.values(), key=len)
    print("longest token:", longest, "len:", len(longest))