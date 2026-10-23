"""
BERT NER training with correct subword-to-char alignment.
Each char gets BIO label; first subword inherits, rest get -100.
"""
import json, os, sys, torch, random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
os.chdir("/dataset/wanggenyi/patient_phex")

random.seed(42); np.random.seed(42); torch.manual_seed(42)

LABELS = ["O", "B-PHEN", "I-PHEN"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

def char_to_bio(text, entities):
    """Convert char-level entity spans to BIO labels per char."""
    labels = ["O"] * len(text)
    for e in entities:
        if e.get("note") == "NO": continue
        off = e["offset"]; ln = e["length"]
        if off < 0 or off + ln > len(text): continue
        labels[off] = "B-PHEN"
        for i in range(off+1, off+ln):
            if labels[i] == "O":
                labels[i] = "I-PHEN"
    return labels

def tokenize_and_align(text, char_labels, tokenizer, max_len=512):
    """Tokenize and align char labels to subwords."""
    enc = tokenizer(text, truncation=True, max_length=max_len, return_offsets_mapping=True)
    tokens = enc["input_ids"]
    offsets = enc["offset_mapping"]
    subword_labels = []
    for start, end in offsets:
        if start == end:
            subword_labels.append(-100)  # special token
        else:
            subword_labels.append(LABEL2ID[char_labels[start]])
    # truncate if needed
    return {
        "input_ids": tokens,
        "attention_mask": enc["attention_mask"],
        "labels": subword_labels[:len(tokens)],
        "offsets": offsets,
    }

class NERDataset(Dataset):
    def __init__(self, docs, tokenizer, max_len=512):
        self.samples = []
        for d in docs:
            for sec in d["full_text"]:
                text = sec["text"]
                if len(text) < 10: continue
                # Map entities to this section's char positions
                sec_entities = []
                for e in d["entities"]:
                    if e.get("note") == "NO": continue
                    off = e["offset"] - sec["offset"]
                    if off >= 0 and off + e["length"] <= len(text):
                        sec_entities.append({"offset": off, "length": e["length"]})
                char_labels = char_to_bio(text, sec_entities)
                # Split long sections into chunks of max_len with overlap
                for start in range(0, len(text), max_len - 50):
                    end = min(start + max_len, len(text))
                    chunk = text[start:end]
                    chunk_labels = char_labels[start:end]
                    if len(chunk) < 20: continue
                    enc = tokenize_and_align(chunk, chunk_labels, tokenizer, max_len)
                    self.samples.append(enc)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        s = self.samples[i]
        return {
            "input_ids": torch.tensor(s["input_ids"]),
            "attention_mask": torch.tensor(s["attention_mask"]),
            "labels": torch.tensor(s["labels"]),
        }

def collate(batch):
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attn = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        l = b["input_ids"].size(0)
        input_ids[i, :l] = b["input_ids"]
        attn[i, :l] = b["attention_mask"]
        labels[i, :l] = b["labels"]
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/pubmedbert")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--out", default="out/ner/final")
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=len(LABELS)).cuda()

    train_docs = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    # Split train: first 60 for train, last 20 for dev
    random.shuffle(train_docs)
    dev_docs = train_docs[-20:]
    tr_docs = train_docs[:-20]

    train_ds = NERDataset(tr_docs, tokenizer)
    dev_ds = NERDataset(dev_docs, tokenizer)
    print(f"train samples: {len(train_ds)}, dev samples: {len(dev_ds)}", flush=True)

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, collate_fn=collate, num_workers=4, pin_memory=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.bs*2, shuffle=False, collate_fn=collate, num_workers=4)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1*total_steps), total_steps)

    best_f1 = 0
    for ep in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_dl:
            batch = {k: v.cuda() for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            total_loss += out.loss.item()
        # Eval
        model.eval()
        tp=fp=fn=0
        with torch.no_grad():
            for batch in dev_dl:
                labels = batch["labels"]
                batch = {k: v.cuda() for k, v in batch.items() if k != "labels"}
                out = model(**batch)
                pred = out.logits.argmax(-1).cpu()
                for i in range(pred.size(0)):
                    for j in range(pred.size(1)):
                        if labels[i,j] == -100: continue
                        p = pred[i,j].item(); l = labels[i,j].item()
                        if l == LABEL2ID["B-PHEN"]:
                            if p == LABEL2ID["B-PHEN"]: tp += 1
                            else: fn += 1
                        elif p == LABEL2ID["B-PHEN"]:
                            fp += 1
        prec = tp/(tp+fp) if tp+fp>0 else 0
        rec = tp/(tp+fn) if tp+fn>0 else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
        print(f"ep{ep} loss={total_loss/len(train_dl):.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)
            print(f"  saved best F1={f1:.4f}", flush=True)
    print(f"best F1={best_f1:.4f}", flush=True)

if __name__ == "__main__":
    main()
