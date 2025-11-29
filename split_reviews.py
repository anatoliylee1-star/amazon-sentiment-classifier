import glob
import os

# Where the extracted dataset is (adjust if needed)
SOURCE_DIR = "sorted_data_acl"

# Where to save split reviews
POS_DIR = "data/raw/amazon/pos"
NEG_DIR = "data/raw/amazon/neg"

os.makedirs(POS_DIR, exist_ok=True)
os.makedirs(NEG_DIR, exist_ok=True)

def split_reviews(review_file, out_dir):
    """Split reviews separated by <review_text> tags."""
    with open(review_file, "r", encoding="latin-1") as f:
        text = f.read()

    reviews = text.split("<review_text>")[1:]  # skip header

    for i, review in enumerate(reviews, start=1):
        clean = review.strip().replace("</review_text>", "")
        filename = os.path.join(out_dir, f"{os.path.basename(review_file)}_{i}.txt")
        with open(filename, "w", encoding="utf-8") as out:
            out.write(clean)

# Process all domains (books, dvd, electronics, etc.)
for domain in glob.glob(os.path.join(SOURCE_DIR, "*")):
    if os.path.isdir(domain):
        pos_file = os.path.join(domain, "positive.review")
        neg_file = os.path.join(domain, "negative.review")

        if os.path.exists(pos_file):
            split_reviews(pos_file, POS_DIR)

        if os.path.exists(neg_file):
            split_reviews(neg_file, NEG_DIR)

print("Done! Reviews split into individual .txt files.")