import re
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

class SpamDetector:
    def __init__(self, model_name: str = "mrm8488/bert-tiny-finetuned-sms-spam-detection", device: int = -1):
        print("[INFO] Loading spam classifier model once...")
        self.pipe = pipeline(
            "text-classification",
            model=model_name,
            device=device,
            truncation=True
        )
        print("[INFO] Model loaded successfully.")

    # ---------- Fast rules check ----------
    @staticmethod
    def fast_spam_check(text: str) -> bool:
        if not text or not isinstance(text, str) or text.strip() == "":
            return True

        t = text.lower().strip()
        words = t.split()

        if len(t) > 500:
            return True
        if re.search(r"(http|www|\.com|\.net|\.org|\.info|\.xyz)", t):
            return True
        if words and max(words.count(w) for w in set(words)) > 10:
            return True
        if len(t) > 20 and sum(c.isupper() for c in t) / len(t) > 0.8:
            return True
        if len(t) > 10 and len(set(t)) < 4:
            return True
        if len(words) < 2 and len(t) < 5:
            return True
        if len(t) > 0 and sum(not c.isalnum() for c in t) / len(t) > 0.8:
            return True
        if len(words) > 10 and len(set(words)) < 3:
            return True
        if t.count("@") > 5:
            return True

        return False

    # ---------- Batch ML prediction ----------
    def classify_batch(self, texts: list):
        preds = self.pipe(texts, batch_size=64, truncation=True, max_length=256)
        labels, scores = [], []
        for pred in preds:
            if pred["label"] in ["LABEL_1", "Spam"]:
                labels.append(1)
                scores.append(pred["score"])
            else:
                labels.append(0)
                scores.append(1 - pred["score"])
        return labels, scores

    # ---------- Main runner ----------
    def spam_detector_pipeline(self, df: pd.DataFrame, text_col: str = "textOriginal", chunk_size: int = 5000) -> pd.DataFrame:
        df = df.copy()
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in dataframe.")

        # Step 1: Rule-based spam tagging
        print("[INFO] Running fast rule-based spam check...")
        rule_flags = df[text_col].fillna("").astype(str).apply(self.fast_spam_check)
        df.loc[rule_flags, "is_spam"] = 1
        df.loc[rule_flags, "spam_score"] = 1.0
        df.loc[~rule_flags, "is_spam"] = None
        df.loc[~rule_flags, "spam_score"] = None
        print(f"[INFO] Rule-based tagged spam: {rule_flags.sum()} / {len(df)}")

        # Step 2: ML prediction for untagged rows
        to_predict_idx = df[df["is_spam"].isna()].index
        print(f"[INFO] Sending {len(to_predict_idx)} rows to ML model")

        n_chunks = (len(to_predict_idx) // chunk_size) + 1
        for i in tqdm(range(n_chunks), desc="ML classification"):
            idx_batch = to_predict_idx[i*chunk_size:(i+1)*chunk_size]
            if len(idx_batch) == 0:
                continue
            texts = df.loc[idx_batch, text_col].fillna("").astype(str).tolist()
            try:
                labels, scores = self.classify_batch(texts)
                df.loc[idx_batch, "is_spam"] = labels
                df.loc[idx_batch, "spam_score"] = scores
            except Exception as e:
                print(f"[ERROR] Batch {i+1} failed: {e}")
                df.loc[idx_batch, "is_spam"] = 0
                df.loc[idx_batch, "spam_score"] = 0.0

        # Step 3: Optional final threshold
        df["is_spam"] = df["spam_score"].apply(lambda x: 1 if x >= 0.2 else 0)

        print(f"[INFO] Completed. Total rows: {len(df)}, Total spam: {df['is_spam'].sum()}")
        return df
