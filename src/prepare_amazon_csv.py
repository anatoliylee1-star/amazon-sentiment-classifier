# src/prepare_amazon_csv.py
import re, random
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_POS = Path("data/raw/amazon/pos")
RAW_NEG = Path("data/raw/amazon/neg")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TOKENS = 5  # outlier removal

def read_folder(folder: Path, label: int):
    rows = []
    if not folder.exists():
        return rows
    for p in folder.glob("*.txt"):
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        text = re.sub(r"\s+", " ", text)
        if len(text.split()) >= MIN_TOKENS:
            rows.append({"text": text, "label": label})
    return rows

def main():
    random.seed(42)
    pos = read_folder(RAW_POS, 1)
    neg = read_folder(RAW_NEG, 0)
    data = pos + neg
    if not data:
        raise SystemExit("No data found under data/raw/amazon/pos and neg")

    df = pd.DataFrame(data).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Count per class
    counts = df["label"].value_counts().to_dict()
    min_class = min(counts.get(0, 0), counts.get(1, 0))

    # Strategy:
    # - If each class has at least 4 samples, do 80/10/10 with stratify on both splits.
    # - If tiny (e.g., 1-3 per class), fall back to 75/25 train/test only with stratify,
    #   and duplicate test.csv as val.csv so training scripts still run.
    if min_class >= 4 and len(df) >= 20:
        # 80/10/10 (stratified)
        train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
        val_df, test_df   = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)
    else:
        # Robust fallback for very small datasets
        print("Small dataset detected -> using 75/25 train/test (stratified) and copying test -> val")
        train_df, test_df = train_test_split(df, test_size=0.25, stratify=df["label"], random_state=42)
        val_df = test_df.copy()

    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)

    print(f"Saved: {OUT_DIR/'train.csv'}, {OUT_DIR/'val.csv'}, {OUT_DIR/'test.csv'}")
    print("Class counts:",
          "\n train:", train_df["label"].value_counts().to_dict(),
          "\n val:",   val_df["label"].value_counts().to_dict(),
          "\n test:",  test_df["label"].value_counts().to_dict())

if __name__ == "__main__":
    main()
