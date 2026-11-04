import random
import numpy as np

from cs336_basics.src.bpe_tokenizer import BPETokenizer, Tokenizer

def sample_docs(path: str, n: int = 10, sep: str = "<|endoftext|>", seed: int = 0) -> list[str]:
    # 只读文件前 ~200MB，避免把整个大文件载入内存
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read(200_000_000)
    docs = [d.strip() for d in text.split(sep) if d.strip()]
    docs = docs[:-1]                      # 丢掉最后一个可能被截断的文档
    random.seed(seed)
    return random.sample(docs, n)

def compression_ratio(tok: Tokenizer, docs: list[str]) -> float:
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    total_tokens = sum(len(tok.encode(d)) for d in docs)
    return total_bytes / total_tokens

if __name__ == "__main__":
    configs = {
        "TinyStories (10K)": ("../data/TinyStoriesV2-GPT4-train.txt",
                              "../data/tinystories_vocab.pkl", "../data/tinystories_merges.pkl"),
        # "OpenWebText (32K)": ("data/owt_train.txt",
        #                       "data/owt_vocab.pkl", "data/owt_merges.pkl"),
    }
    for name, (data, vp, mp) in configs.items():
        tok = Tokenizer.from_files(vp, mp, special_tokens=["<|endoftext|>"])
        docs = sample_docs(data)
        print(f"{name}: {compression_ratio(tok, docs):.2f} bytes/token")