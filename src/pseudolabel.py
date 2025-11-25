import os, csv, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

UNLAB_TXT = "data/raw/amazon/unlabeled/book_unlabeled.txt"
OUT_CSV   = "data/processed/pseudolabeled.csv"
MODEL_DIR = os.getenv("LOCAL_MODEL_PATH", "models/distilbert_best")
MAX_LEN   = 256
THRESH    = 0.90  # keep only very confident predictions

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()

    kept = 0
    with open(UNLAB_TXT, "r", encoding="utf-8", errors="ignore") as f_in, \
         open(OUT_CSV, "w", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out)
        w.writerow(["text","label"])  # label: 1=pos, 0=neg
        for line in f_in:
            text = line.strip()
            if not text:
                continue
            enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN).to(device)
            with torch.inference_mode():
                logits = model(**enc).logits.softmax(-1).squeeze(0)
            conf, label_id = float(torch.max(logits, dim=-1).values), int(torch.argmax(logits))
            if conf >= THRESH:
                w.writerow([text, label_id])
                kept += 1
    print(f"✅ Saved {kept} high-confidence pseudo-labeled rows → {OUT_CSV}")

if __name__ == "__main__":
    main()

