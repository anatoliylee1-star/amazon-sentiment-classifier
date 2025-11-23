import math
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/distilbert_best")
PRETRAINED = "distilbert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
WARMUP_RATIO = 0.06

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")      # Apple Silicon
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextDataset(Dataset):
    def __init__(self, df, tok):
        self.df = df.reset_index(drop=True)
        self.tok = tok
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        enc = self.tok(row["text"], truncation=True, padding="max_length",
                       max_length=MAX_LEN, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    preds, labels = [], []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        p = torch.argmax(logits, dim=-1).cpu().tolist()
        y = batch["labels"].cpu().tolist()
        preds.extend(p); labels.extend(y)
    acc = accuracy_score(labels, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return acc, pr, rc, f1

def main():
    print("🔧 Loading data from", DATA_DIR.resolve())
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df   = pd.read_csv(DATA_DIR / "val.csv")
    test_df  = pd.read_csv(DATA_DIR / "test.csv")
    print("Counts:", len(train_df), len(val_df), len(test_df))

    print("🧠 Loading tokenizer/model:", PRETRAINED)
    tok = AutoTokenizer.from_pretrained(PRETRAINED)
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED, num_labels=2)

    device = get_device()
    model.to(device)
    print("💻 Device:", device)

    train_ds = TextDataset(train_df, tok)
    val_ds   = TextDataset(val_df, tok)
    test_ds  = TextDataset(test_df, tok)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    optim = AdamW(model.parameters(), lr=LR)
    steps_total = len(train_dl) * EPOCHS
    steps_warm  = max(1, int(WARMUP_RATIO * steps_total))
    sched = get_linear_schedule_with_warmup(optim, steps_warm, steps_total)

    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        print(f"\n🚀 Epoch {epoch}/{EPOCHS}")
        model.train()
        loop = tqdm(train_dl, desc=f"train {epoch}", leave=False)
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step(); sched.step(); optim.zero_grad()
            loop.set_postfix(loss=float(loss.item()))

        acc, pr, rc, f1 = evaluate(model, val_dl, device)
        print(f"[val] acc={acc:.4f} pr={pr:.4f} rc={rc:.4f} f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tok.save_pretrained(MODEL_DIR)
            print(f"✅ Saved best to {MODEL_DIR} (f1={best_f1:.4f})")

    # Final test with best model
    best = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    acc, pr, rc, f1 = evaluate(best, test_dl, device)
    print(f"[test] acc={acc:.4f} pr={pr:.4f} rc={rc:.4f} f1={f1:.4f}")

if __name__ == "__main__":
    main()

