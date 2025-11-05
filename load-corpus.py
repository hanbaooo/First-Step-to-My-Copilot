from datasets import load_dataset

def save_small_subset(save_path="corpus_small.txt", num_lines=10000, split="train"):
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    texts = dataset["text"]
    filtered = [t for t in texts if t and not t.isspace()]
    subset = filtered[:num_lines]
    corpus = "\n".join(subset)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"Saved {num_lines} lines to {save_path}, ~{len(corpus):,} chars")

if __name__ == "__main__":
    save_small_subset(save_path="corpus.txt", num_lines=5000)

