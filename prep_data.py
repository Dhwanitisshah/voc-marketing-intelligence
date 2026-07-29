import pandas as pd

RAW = r"C:\Users\Dhwanit Shah\Downloads\archive (2)\1429_1.csv"   # <-- change to your real path
df = pd.read_csv(RAW, low_memory=False)

df = df.rename(columns={
    "reviews.text": "review_text",
    "reviews.rating": "rating",
    "reviews.date": "date",
})
df = df[["review_text", "rating", "date"]].dropna(subset=["review_text"])
df["review_text"] = df["review_text"].astype(str).str.strip()
df = df[df["review_text"].str.len() > 0]
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
print("clean rows:", len(df))

df.sample(min(3000, len(df)), random_state=42).to_csv("data/raw/reviews.csv", index=False)
df.sample(min(400, len(df)), random_state=1).to_csv("data/raw/reviews_400.csv", index=False)
print("wrote data/raw/reviews.csv (3000) and reviews_400.csv (400)")