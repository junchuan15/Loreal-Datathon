import pandas as pd
import zipfile
import re
import emoji
import ast
from datetime import timedelta
from typing import List, Optional, Union

class DataIngestionPipeline:
    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        self.comments_df = None
        self.videos_df = None
        self.merged_df = None

        # ---------- Regex Patterns ----------
        self.PATTERNS = {
            "hashtag": re.compile(r"#(\w+)"),
            "emoji": re.compile(
                "[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|[\U0001F1E0-\U0001F1FF]",
                flags=re.UNICODE
            ),
            "url": re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE),
            "mention": re.compile(r"@\w+", re.IGNORECASE),
            "special_char": re.compile(r"[^\w\s#]", re.UNICODE),
            "space": re.compile(r"\s+")
        }

    # ------------------- Load Data -------------------
    def load_zip_data(self):
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            file_list = z.namelist()
            comments_files = [f for f in file_list if f.lower().startswith("comments")]
            videos_files = [f for f in file_list if "videos" in f.lower()]
            
            comments_dfs = [pd.read_csv(z.open(f)) for f in comments_files]
            self.comments_df = pd.concat(comments_dfs, ignore_index=True)
            
            if videos_files:
                self.videos_df = pd.read_csv(z.open(videos_files[0]))
            else:
                raise ValueError("No video file found in zip.")

    # ------------------- Helper Methods -------------------
    def _clean_token(self, tok: str) -> Union[str, None]:
        if not isinstance(tok, str):
            return None
        t = tok.strip().lower()
        t = re.sub(r"[^a-z0-9_]", "", t)
        if not t or t.isnumeric() or len(t) <= 3:
            return None
        return t

    def extract_hashtags(self, value) -> List[str]:
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
            c = self._clean_token(tok)
            if c and c not in seen:
                cleaned.append(c)
                seen.add(c)
        return cleaned

    def extract_emojis(self, text: str) -> List[str]:
        if not isinstance(text, str):
            return []
        emojis = list(set(self.PATTERNS["emoji"].findall(text)))
        return [emoji.demojize(e, delimiters=(" ", " ")).strip() for e in emojis]

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = self.PATTERNS["url"].sub(" ", text)
        text = self.PATTERNS["mention"].sub(" ", text)
        text = self.PATTERNS["special_char"].sub(" ", text)
        text = self.PATTERNS["space"].sub(" ", text).strip()
        text = self.PATTERNS["hashtag"].sub(" ", text)
        text = self.PATTERNS["emoji"].sub(" ", text)
        text = " ".join(w for w in text.split() if re.fullmatch(r"[a-z0-9]+", w))
        return text

    def iso8601_to_seconds(self, duration: str) -> int:
        if not isinstance(duration, str) or not duration.startswith("P"):
            return None
        time_str = duration.replace("P", "")
        days, hours, minutes, seconds = 0, 0, 0, 0
        day_match = re.search(r"(\d+)D", time_str)
        if day_match:
            days = int(day_match.group(1))
        time_part = time_str.split("T")[-1] if "T" in time_str else time_str
        hour_match = re.search(r"(\d+)H", time_part)
        minute_match = re.search(r"(\d+)M", time_part)
        second_match = re.search(r"(\d+)S", time_part)
        if hour_match:
            hours = int(hour_match.group(1))
        if minute_match:
            minutes = int(minute_match.group(1))
        if second_match:
            seconds = int(second_match.group(1))
        return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds).total_seconds()

    def extract_topic_categories(self, cat_str: str) -> List[str]:
        if not isinstance(cat_str, str):
            return []
        try:
            cats = ast.literal_eval(cat_str) if cat_str.startswith("[") else [cat_str]
        except Exception:
            cats = [cat_str]
        return [c.split("wiki/")[-1] for c in cats if "wiki/" in c]

    # ------------------- Preprocess Videos -------------------
    def preprocess_videos(self):
        if self.videos_df is None:
            raise ValueError("Videos dataframe is empty.")

        out = self.videos_df.copy()

        if "publishedAt" in out.columns:
            out["publishedAt"] = pd.to_datetime(out["publishedAt"], errors="coerce").dt.date
            cutoff = pd.to_datetime("2025-01-01").date()
            out = out[out["publishedAt"] >= cutoff]

        if "contentDuration" in out.columns:
            out["contentDuration (seconds)"] = out["contentDuration"].apply(self.iso8601_to_seconds)

        for col in ["viewCount", "likeCount", "favouriteCount", "commentCount"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

        if "videoId" in out.columns:
            out["videoId"] = out["videoId"].astype(str).apply(lambda x: f"V_{x}")
        if "channelId" in out.columns:
            out["channelId"] = out["channelId"].astype(str).apply(lambda x: f"CH_{x}")

        if "topicCategories" in out.columns:
            out["topicCategories"] = out["topicCategories"].apply(self.extract_topic_categories)

        title = out["title"].fillna("") if "title" in out.columns else ""
        desc = out["description"].fillna("") if "description" in out.columns else ""
        combined = title + desc
        out["hashtags_video"] = combined.apply(self.extract_hashtags)
        out["emojis_video"] = combined.apply(self.extract_emojis)
        out["video_text"] = combined.apply(self.clean_text)

        out["engagement_rate"] = (out["likeCount"] + out["commentCount"]) / out["viewCount"].replace(0, 1)

        self.videos_df = out

    # ------------------- Preprocess Comments -------------------
    def preprocess_comments(self):
        if self.comments_df is None:
            raise ValueError("Comments dataframe is empty.")
        out = self.comments_df.copy()

        if "commentId" in out.columns:
            out["commentId"] = out["commentId"].apply(lambda x: f"CM_{x}")
        if "channelId" in out.columns:
            out["channelId"] = out["channelId"].astype(str).apply(lambda x: f"CH_{x}")
        if "videoId" in out.columns:
            out["videoId"] = out["videoId"].apply(lambda x: f"V_{x}")
        if "authorId" in out.columns:
            out["authorId"] = out["authorId"].apply(lambda x: f"ATH_{x}")
        if "parentCommentId" in out.columns:
            out["parentCommentId"] = (
                out["parentCommentId"]
                .fillna(0)
                .astype(int)
                .apply(lambda x: f"PC_{x}" if x != 0 else "")
            )
        if "textOriginal" in out.columns:
            out["textOriginal"] = out["textOriginal"].fillna("")
        for col in ["publishedAt", "updatedAt"]:
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], errors="coerce").dt.date
        if "publishedAt" in out.columns:
            cutoff = pd.to_datetime("2025-01-01").date()
            out = out[out["publishedAt"] >= cutoff]

        self.comments_df = out

    # ------------------- Merge -------------------
    def merge_data(self):
        if self.comments_df is None or self.videos_df is None:
            raise ValueError("Dataframes must be loaded and preprocessed first.")
        self.merged_df = self.comments_df.merge(
            self.videos_df,
            on=["videoId", "channelId"],
            how="inner",
            suffixes=("_comment", "_video")
        )

    # ------------------- Full Pipeline -------------------
    def ingest_pipeline(self):
        self.load_zip_data()
        self.preprocess_videos()
        self.preprocess_comments()
        self.merge_data()
        return self.merged_df
