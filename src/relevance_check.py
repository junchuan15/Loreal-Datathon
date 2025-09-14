import re
import emoji
from typing import List, Union
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

PATTERNS = {
    "hashtag": re.compile(r"#(\w+)"),
    "emoji": re.compile(
        "[\U0001F600-\U0001F64F]"     
        "|[\U0001F300-\U0001F5FF]"    
        "|[\U0001F680-\U0001F6FF]"    
        "|[\U0001F1E0-\U0001F1FF]",    
        flags=re.UNICODE
    ),
    "url": re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE),
    "mention": re.compile(r"@\w+", re.IGNORECASE),
    "special_char": re.compile(r"[^\w\s#]", re.UNICODE),
    "space": re.compile(r"\s+")
}

class RelevanceChecker:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', device: int = -1):
        print("[INFO] Loading embedding model...")
        self.embed_model = SentenceTransformer(model_name, device=device)
        print("[INFO] Model loaded.")

    # ---------- Helper methods ----------
    @staticmethod
    def _clean_token(tok: str) -> Union[str, None]:
        if not isinstance(tok, str):
            return None
        t = tok.strip().lower()
        t = re.sub(r"[^a-z0-9_]", "", t)
        if not t or t.isnumeric() or len(t) <= 3:
            return None
        return t

    @staticmethod
    def extract_hashtags(value) -> List[str]:
        if isinstance(value, list):
            tokens = [str(x) for x in value if x is not None]
        elif isinstance(value, str):
            s = value.strip()
            if s == "":
                return []
            if "#" in s:
                tokens = re.findall(r"#(\w+)", s.lower())
            else:
                if "," in s:
                    parts = s.split(",")
                elif "|" in s:
                    parts = s.split("|")
                else:
                    parts = s.split()
                tokens = [p.strip() for p in parts if p.strip()]
        else:
            return []

        cleaned = []
        seen = set()
        for tok in tokens:
            c = RelevanceChecker._clean_token(tok)
            if c and c not in seen:
                cleaned.append(c)
                seen.add(c)
        return cleaned

    @staticmethod
    def extract_emojis(text: str) -> List[str]:
        if not isinstance(text, str):
            return []
        emojis = list(set(PATTERNS["emoji"].findall(text)))
        return [emoji.demojize(e, delimiters=(" ", " ")).strip() for e in emojis]

    @staticmethod
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = PATTERNS["url"].sub(" ", text)
        text = PATTERNS["mention"].sub(" ", text)
        text = PATTERNS["special_char"].sub(" ", text)
        text = PATTERNS["space"].sub(" ", text).strip()
        text = PATTERNS["hashtag"].sub(" ", text)
        text = PATTERNS["emoji"].sub(" ", text)
        text = " ".join(w for w in text.split() if re.fullmatch(r"[a-z0-9]+", w))
        return text

    # ---------- Preprocessing ----------
    def preprocess_comments(self, df: pd.DataFrame, text_col: str = "textOriginal") -> pd.DataFrame:
        df = df.copy()
        df["hashtags_comments"] = df[text_col].apply(self.extract_hashtags)
        df["emojis_comments"] = df[text_col].apply(self.extract_emojis)
        df["comment_clean"] = df[text_col].apply(self.clean_text)
        return df

    # ---------- Video embeddings ----------
    def precompute_video_embeddings(self, df: pd.DataFrame) -> dict:
        unique_videos = df[['videoId', 'video_text']].drop_duplicates()
        video_embeddings_dict = {}
        for _, row in tqdm(unique_videos.iterrows(), total=len(unique_videos), desc="Precomputing video embeddings"):
            vid = row['videoId']
            base_text = row['video_text'] if isinstance(row['video_text'], str) else ""
            if not base_text.strip():
                continue
            video_embeddings_dict[vid] = self.embed_model.encode(base_text, convert_to_tensor=True)
        return video_embeddings_dict

    # ---------- Compute weighted relevance ----------
    @staticmethod
    def compute_relevance_weight(comment_embeddings, video_embeddings, c_hashtags, v_hashtags,
                                 sem_weight=1.0, hashtag_bonus=0.10, max_score=1.0):
        sem_scores = util.cos_sim(comment_embeddings, video_embeddings).diagonal().cpu().numpy()
        bonus_scores = [hashtag_bonus if ch and vh and len(set(ch) & set(vh)) > 0 else 0.0
                        for ch, vh in zip(c_hashtags, v_hashtags)]
        final_scores = [min(sem_weight*s + b, max_score) for s, b in zip(sem_scores, bonus_scores)]
        return final_scores

    # ---------- Main relevance pipeline ----------
    def relevance_check_pipeline(self, df: pd.DataFrame,
            sem_weight=1.0,
            hashtag_bonus=0.10,
            max_score=1.0,
            threshold=0.4,
            batch_size=64,
            chunk_size=5000) -> pd.DataFrame:

        df = df.copy()
        df["weighted_relevance"] = 0.0
        df["is_relevant"] = 0

        video_embeddings_dict = self.precompute_video_embeddings(df)
        if not video_embeddings_dict:
            print("[WARN] No video embeddings found.")
            return df
        vids_with_emb = set(video_embeddings_dict.keys())

        n_chunks = (len(df) // chunk_size) + 1
        for i in range(n_chunks):
            start, end = i*chunk_size, (i+1)*chunk_size
            chunk_df = df.iloc[start:end]
            if chunk_df.empty:
                continue

            non_empty_df = chunk_df[chunk_df["comment_clean"].notna() & (chunk_df["comment_clean"].str.strip() != "")]
            empty_df = chunk_df.drop(non_empty_df.index)
            if not non_empty_df.empty:
                has_emb_mask = non_empty_df["videoId"].isin(vids_with_emb)
                proc_df = non_empty_df[has_emb_mask]
                no_embed_df = non_empty_df[~has_emb_mask]
            else:
                proc_df, no_embed_df = non_empty_df.copy(), non_empty_df.copy()

            if not no_embed_df.empty:
                df.loc[no_embed_df.index, ["weighted_relevance", "is_relevant"]] = [0.0, 0]
            if proc_df.empty:
                if not empty_df.empty:
                    df.loc[empty_df.index, ["weighted_relevance", "is_relevant"]] = [0.0, 0]
                continue

            comments = proc_df['comment_clean'].tolist()
            videos = proc_df['videoId'].tolist()
            c_hashtags = [x if isinstance(x, list) else [] for x in proc_df['hashtags_comments'].tolist()]
            v_hashtags = [x if isinstance(x, list) else [] for x in proc_df['hashtags_video'].tolist()]

            comment_embeddings_list = []
            for j in tqdm(range(0, len(comments), batch_size), desc=f"Chunk {i+1} batches"):
                batch_emb = self.embed_model.encode(comments[j:j+batch_size], convert_to_tensor=True)
                comment_embeddings_list.append(batch_emb)
            comment_embeddings = torch.cat(comment_embeddings_list)
            video_embeddings = torch.stack([video_embeddings_dict[vid] for vid in videos])
            weighted_scores = self.compute_relevance_weight(comment_embeddings, video_embeddings,
                                                            c_hashtags, v_hashtags,
                                                            sem_weight, hashtag_bonus, max_score)
            df.loc[proc_df.index, "weighted_relevance"] = weighted_scores
            df.loc[proc_df.index, "is_relevant"] = (pd.Series(weighted_scores, index=proc_df.index) >= threshold).astype(int)

            if not empty_df.empty:
                df.loc[empty_df.index, ["weighted_relevance", "is_relevant"]] = [0.0, 0]

        return df
