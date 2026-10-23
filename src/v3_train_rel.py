"""
Train PubMedBERT-based relation classifier with 4-card DDP.
"""
import os, json, argparse, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

class RelDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=256):
        self.data = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        s = self.data[i]
        enc = self.tok(s["text"], truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(s["label"], dtype=torch.long),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out", default="out/rel/final")
    args = ap.parse_args()

    init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    train_ds = RelDataset("data/rel/train.jsonl", tokenizer)
    dev_ds = RelDataset("data/rel/dev.jsonl", tokenizer)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
    train_dl = DataLoader(train_ds, batch_size=args.bs, sampler=train_sampler, num_workers=4, pin_memory=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.bs*2, shuffle=False, num_workers=4)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1*total_steps), total_steps)

    best_f1 = 0
    for ep in range(args.epochs):
        train_sampler.set_epoch(ep)
        model.train()
        total_loss = 0
        for batch in train_dl:
            batch = {k: v.cuda(local_rank) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            total_loss += out.loss.item()
        # eval
        model.eval()
        tp=fp=fn=0
        with torch.no_grad():
            for batch in dev_dl:
                batch = {k: v.cuda(local_rank) for k, v in batch.items()}
                out = model(**batch)
                pred = out.logits.argmax(-1)
                for p, l in zip(pred, batch["labels"]):
                    p=p.item(); l=l.item()
                    if p==1 and l==1: tp+=1
                    elif p==1 and l==0: fp+=1
                    elif p==0 and l==1: fn+=1
        prec = tp/(tp+fp) if tp+fp>0 else 0
        rec = tp/(tp+fn) if tp+fn>0 else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
        if rank==0:
            print(f"ep{ep} loss={total_loss/len(train_dl):.4f} dev P={prec:.4f} R={rec:.4f} F1={f1:.4f}", flush=True)
            if f1 > best_f1:
                best_f1 = f1
                model.module.save_pretrained(args.out)
                tokenizer.save_pretrained(args.out)
                print(f"  saved best F1={f1:.4f}", flush=True)
    if rank==0:
        print(f"best dev F1={best_f1:.4f}", flush=True)
    destroy_process_group()

if __name__ == "__main__":
    main()
