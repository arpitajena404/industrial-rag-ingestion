import json, random

with open("data/processed/chunks.jsonl", encoding="utf-8") as f:
    lines = f.readlines()

for line in random.sample(lines, 5):
    obj = json.loads(line)
    print("TEXT:", obj["text"][:200])
    print("META:", {k: v for k, v in obj.items() if k != "text"})
    print("---")