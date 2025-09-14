# text_clustering_fullpipeline.py

import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Ensure required NLTK resources
resources = [
    "punkt",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger"
]
for r in resources:
    try:
        nltk.data.find(r)
    except LookupError:
        nltk.download(r, quiet=True)

class ProductClustering:
    """
    Full pipeline based on the code you provided.
    Usage:
        pipeline = TextClusteringPipeline(input_df)
        result_df = pipeline.clustering_pipeline()   # returns dataframe with cluster_label column
    """

    def __init__(self, input_df: pd.DataFrame):
        if "textOriginal" not in input_df.columns:
            raise ValueError("Input dataframe must have a 'textOriginal' column")
        # keep a copy of original dataframe
        self.df = input_df.copy()
        # core models / holders
        self.model = None
        self.kw_model = None
        self.embeddings = None
        self.optimal_k = None
        self.cluster_labels = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

        # helpers
        self.tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        self.spell = SpellChecker(distance=1)
        self.stop_words = set(stopwords.words("english"))
        self.stop_words.update(self._custom_stopwords())
        self.lemmatizer = WordNetLemmatizer()

        # mapping dictionaries (kept as you provided)
        self.cluster_keywords_mapping = {
            'Noise/Outliers': ['beautiful', 'hair', 'makeup', 'laugh', 'loud', 'laugh loud', 'girl', 'brother'],
            'Curly & Hair': ['curly', 'hair', 'curly hair', 'curl', 'wavy', 'hair curly', 'straight', 'wave'],
            'Que & Linda': ['que', 'la', 'linda', 'el', 'hermosa', 'le', 'belle', 'como'],
            'Hair & Cut': ['hair', 'cut', 'blonde', 'beautiful', 'short', 'girl', 'color', 'natural'],
            'Skin & Color': ['skin', 'color', 'beautiful', 'brown', 'face', 'skin beautiful', 'shade', 'sunscreen'],
            'Map & Cantik': ['ya', 'map', 'cantik', 'ke', 'kayak', 'bang', 'ko', 'makeup'],
            'Beautiful & Girl': ['beautiful', 'girl', 'natural', 'gorgeous', 'god', 'beauty', 'face', 'woman'],
            'Makeup & Beautiful': ['makeup', 'beautiful', 'beautiful makeup', 'skin', 'wear makeup', 'wear', 'makeup beautiful', 'girl'],
            'Eye & Lip': ['eye', 'lip', 'lipstick', 'eyebrow', 'lash', 'eyelash', 'foundation', 'mascara']
        }

        self.cluster_name_mapping = {
            'Beautiful & Girl': 'Physical appearance',
            'Makeup & Beautiful': 'Make Up',
            'Skin & Color': 'Skin care',
            'Noise/Outliers': 'others',
            'Hair & Cut': 'Hair treatment',
            'Curly & Hair': 'Hair treatment',
            'Map & Cantik': 'others',
            'Eye & Lip': 'Cosmetics (Eye & Lip)',
            'Que & Linda': 'others'
        }

    # ---------------- helper functions (preprocessing) ----------------
    def _basic_cleaning(self, text):
        if pd.isna(text) or text == "":
            return ""
        text = text.replace("\\n", " ").replace("\\t", " ")
        text = text.lower()
        text = re.sub(r"[@#]\w+", "", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\b[^aeiou\s]{6,}\b", "", text)
        text = re.sub(r"\b[a-z0-9]\b", "", text)
        text = re.sub(r"\b\d+\b", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _remove_emojis(self, tokens):
        emoji_pattern = re.compile("["
                                   "\U0001F600-\U0001F64F"
                                   "\U0001F300-\U0001F5FF"
                                   "\U0001F680-\U0001F6FF"
                                   "\U0001F1E0-\U0001F1FF"
                                   "\U00002702-\U000027B0"
                                   "\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return [emoji_pattern.sub(r'', t) for t in tokens if emoji_pattern.sub(r'', t)]

    def _expand_slang(self, tokens):
        slang_dict = {
            "bc": "because", "ur": "your", "u": "you", "r": "are", "n": "and",
            "w/": "with", "b4": "before", "2": "to", "4": "for", "luv": "love",
            "plz": "please", "thx": "thanks", "omg": "oh my god", "lol": "laugh out loud",
            "brb": "be right back", "ttyl": "talk to you later", "idk": "i do not know",
            "tbh": "to be honest", "imo": "in my opinion", "imho": "in my humble opinion",
            "fyi": "for your information", "aka": "also known as", "asap": "as soon as possible",
            "btw": "by the way", "eww": "disgusting",
        }
        expanded = []
        for token in tokens:
            expanded_token = slang_dict.get(token.lower(), token)
            if " " in expanded_token:
                expanded.extend(expanded_token.split())
            else:
                expanded.append(expanded_token)
        return expanded

    def _remove_short_tokens(self, tokens, min_length=2):
        keep_short = {"i","a","is","it","me","my","we","he","be","do","so","no","up","on","in","at","to","of","or"}
        return [t for t in tokens if len(t) >= min_length or t in keep_short]

    def _correct_spelling_fast(self, tokens):
        corrected = []
        for t in tokens:
            if len(t) > 2:
                cor = self.spell.correction(t)
                corrected.append(cor if cor else t)
            else:
                corrected.append(t)
        return corrected

    def _remove_stopwords(self, tokens):
        return [t for t in tokens if t.lower() not in self.stop_words]

    def _remove_non_meaningful(self, tokens):
        cleaned = []
        for t in tokens:
            if t in string.punctuation: continue
            if t.isdigit(): continue
            if len(t) == 1 and t not in ["i","a"]: continue
            if re.match(r"^[^\w\s]+$", t): continue
            cleaned.append(t)
        return cleaned

    def _normalize_tokens(self, tokens):
        def get_wordnet_pos(tag):
            if tag.startswith("J"): return wordnet.ADJ
            elif tag.startswith("V"): return wordnet.VERB
            elif tag.startswith("N"): return wordnet.NOUN
            elif tag.startswith("R"): return wordnet.ADV
            else: return wordnet.NOUN

        pos_tags = pos_tag(tokens, lang="eng")
        return [self.lemmatizer.lemmatize(t.lower(), get_wordnet_pos(tag)) for t, tag in pos_tags]

    def _custom_stopwords(self):
        return {
            "like","get","go","would","could","should","also","really","much","many",
            "good","great","love","nice","best","use","using","used","got","going",
            "even","still","just","want","need","make","made","way","time","day",
            "work","works","try","tried","buy","bought","look","looks","please","one",
            "cute","thank","thanks","video","hai","hi","wow","people","think","omg","ur",
            "bro","without","better","see","seen","thing","things","say","said","tell",
            "come","came","back","first","second","last","next","new","old","right",
            "left","yes","yeah","okay","ok","well","maybe","probably","actually","totally",
            "super","pretty","very","quite","definitely","absolutely","completely","name",
            "lol","always","black","real","im","men","videos","never","wig","share","dont",
            "long","stunning","put","thought","song","dear","every","oh","stop","pls"
        }

    # ---------------- TF-IDF filtering (keeps all rows; marks "<empty>") ----------------
    def _apply_tfidf_filtering(self):
        non_empty = self.df[self.df["textcleaned"].str.len() > 0]["textcleaned"].tolist()
        if not non_empty:
            # ensure tfidf vectorizer exists (empty)
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2), min_df=1, max_df=0.8)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform([""])
            return

        # Fit TF-IDF on the non-empty texts (domain weighting)
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2), min_df=1, max_df=0.8)
        tfidf_matrix_local = tfidf_vectorizer.fit_transform(non_empty)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        mean_scores = np.array(tfidf_matrix_local.mean(axis=0)).flatten()
        tfidf_df = pd.DataFrame({"word": feature_names, "tfidf_score": mean_scores})

        # Weight cosmetic/domain terms (as you had)
        cosmetics_terms = {
            "makeup","lipstick","foundation","mascara","eyeliner","eyeshadow","blush",
            "primer","concealer","powder","serum","toner","shampoo","perfume","spf"
        }
        tfidf_df["weighted_score"] = tfidf_df.apply(
            lambda r: r["tfidf_score"] * 5 if r["word"] in cosmetics_terms else r["tfidf_score"],
            axis=1
        )

        # choose high-score words
        high_score_words = set(tfidf_df[tfidf_df["weighted_score"] >= 0.0005]["word"])

        # filter tokens in textcleaned but keep rows (if filtered empty → set "<empty>")
        def filter_tokens(txt):
            filtered = [t for t in txt.split() if t in high_score_words]
            return " ".join(filtered) if filtered else "<empty>"

        self.df["textcleaned"] = self.df["textcleaned"].apply(filter_tokens)

        # store vectorizer & matrix for whole df (transform)
        self.tfidf_vectorizer = tfidf_vectorizer
        # transform the current textcleaned (including "<empty>" rows)
        self.tfidf_matrix = self.tfidf_vectorizer.transform(self.df["textcleaned"].replace("<empty>", ""))

    # ---------------- preprocessing pipeline ----------------
    def preprocess(self):
        print("Preprocessing text...")
        # apply chain of your preprocessing steps; keep rows
        self.df["textcleaned"] = self.df["textOriginal"].apply(self._basic_cleaning)
        self.df["textcleaned"] = self.df["textcleaned"].apply(self.tokenizer.tokenize)
        self.df["textcleaned"] = self.df["textcleaned"].apply(self._remove_emojis)
        self.df["textcleaned"] = self.df["textcleaned"].apply(self._expand_slang)
        self.df["textcleaned"] = self.df["textcleaned"].apply(self._remove_short_tokens)
        self.df["textcleaned"] = self.df["textcleaned"].apply(self._correct_spelling_fast)
        self.df["textcleaned"] = self.df["textcleaned"].apply(self._remove_stopwords)
        self.df["textcleaned"] = self.df["textcleaned"].apply(self._remove_non_meaningful)
        self.df["textcleaned"] = self.df["textcleaned"].apply(self._normalize_tokens)
        self.df["textcleaned"] = self.df["textcleaned"].apply(lambda x: " ".join(x) if isinstance(x, (list,tuple)) else str(x))
        # apply TF-IDF filtering (keeps rows, sets "<empty>" for fully filtered)
        self._apply_tfidf_filtering()

    # ---------------- data helpers ----------------
    def load_data(self):
        print("Loading data...")
        # ensure column exists
        if "textcleaned" not in self.df.columns:
            raise ValueError("Input dataframe must have a 'textcleaned' column; run preprocess() first")

        # fill missing and mark too short (we won't drop)
        self.df["textcleaned"] = self.df["textcleaned"].fillna("<empty>").astype(str)
        self.df["too_short"] = self.df["textcleaned"].str.len() < 10
        print(f"Loaded {len(self.df)} rows (too_short: {self.df['too_short'].sum()})")
        return self.df

    # ---------------- model initialization / embeddings ----------------
    def initialize_models(self):
        print("Initializing models...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.kw_model = KeyBERT(model=self.model)

    def create_embeddings(self):
        print("Creating sentence embeddings...")
        texts = self.df["textcleaned"].tolist()
        # encode all rows; later we cluster only valid ones
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Embeddings shape: {self.embeddings.shape}")

    # ---------------- find k with silhouette ----------------
    def find_optimal_clusters(self, max_k=15):
        print("Finding optimal number of clusters...")
        n = len(self.embeddings)
        max_k_eff = min(max_k, max(2, n // 2))
        k_range = range(2, max_k_eff + 1)
        silhouette_scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            # cluster only non-empty, non-too_short indices
            mask_valid = (~self.df["too_short"]) & (self.df["textcleaned"] != "<empty>")
            if mask_valid.sum() < k:
                silhouette_scores.append(-1)  # can’t evaluate, too few valid points
                print(f"k={k}, skipped (not enough valid rows)")
                continue
            valid_embeddings = self.embeddings[mask_valid.values]
            labels = kmeans.fit_predict(valid_embeddings)
            sil = silhouette_score(valid_embeddings, labels) if len(set(labels)) > 1 else -1
            silhouette_scores.append(sil)
            print(f"k={k}, Silhouette Score: {sil:.4f}")
        optimal_idx = int(np.argmax(silhouette_scores))
        self.optimal_k = list(k_range)[optimal_idx]
        print(f"\nOptimal number of clusters: {self.optimal_k}")
        return self.optimal_k

    # ---------------- perform clustering ----------------
    def perform_clustering(self, k=None):
        print("Performing clustering...")
        if k is None:
            k = self.optimal_k
        # prepare mask of valid rows
        mask_valid = (~self.df["too_short"]) & (self.df["textcleaned"] != "<empty>")
        self.df["cluster_label"] = "other"   # default for invalid rows

        if mask_valid.sum() == 0:
            print("No valid rows to cluster. All rows labeled 'other'.")
            self.cluster_labels = self.df["cluster_label"]
            return self.df["cluster_label"]

        valid_embeddings = self.embeddings[mask_valid.values]
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        valid_labels = kmeans.fit_predict(valid_embeddings)

        # Map labels back into full df
        self.df.loc[mask_valid, "cluster_label"] = valid_labels
        self.cluster_labels = self.df["cluster_label"]
        print("Clustering completed!")
        return self.df["cluster_label"]

    # ---------------- keyword extraction ----------------
    def extract_cluster_keywords(self, top_n=10):
        print("Extracting cluster keywords...")
        # ensure tfidf_vectorizer exists; if not, fit on textcleaned
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df["textcleaned"].replace("<empty>", ""))

        cluster_info = {}
        # iterate only over numeric cluster ids that were assigned
        numeric_ids = sorted([v for v in set(self.df["cluster_label"]) if v != "other"])
        for cluster_id in numeric_ids:
            cluster_texts = self.df[self.df["cluster_label"] == cluster_id]["textcleaned"].tolist()
            cluster_size = len(cluster_texts)
            combined_text = " ".join(cluster_texts)
            # KeyBERT keywords
            try:
                kb = self.kw_model.extract_keywords(
                    combined_text,
                    keyphrase_ngram_range=(1,2),
                    stop_words="english",
                    top_n=top_n,
                    use_mmr=True,
                    diversity=0.5
                )
                keybert_kws = [k[0] for k in kb]
            except Exception:
                keybert_kws = []

            # TF-IDF keywords: compute mean TF-IDF across docs in cluster
            cluster_indices = self.df[self.df["cluster_label"] == cluster_id].index
            # make sure tfidf_matrix has rows aligned to df (transform current textcleaned)
            tfidf_mat_all = self.tfidf_vectorizer.transform(self.df["textcleaned"].replace("<empty>", ""))
            # map cluster_indices to corresponding rows in tfidf_mat_all (since "<empty>" rows were replaced by "")
            # easiest: compute tfidf for cluster_texts directly
            if cluster_texts:
                cluster_tfidf_local = self.tfidf_vectorizer.transform(cluster_texts).mean(axis=0).A1
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                top_idx = cluster_tfidf_local.argsort()[-top_n:][::-1]
                tfidf_keywords = [feature_names[i] for i in top_idx if cluster_tfidf_local[i] > 0]
            else:
                tfidf_keywords = []

            # representative samples (closest to centroid)
            # Build centroid on valid embeddings slice corresponding to cluster rows
            # find indices into embeddings
            mask_cluster = (self.df["cluster_label"] == cluster_id).values
            if mask_cluster.sum() > 0:
                cluster_embeds = self.embeddings[mask_cluster]
                centroid = cluster_embeds.mean(axis=0)
                distances = np.linalg.norm(cluster_embeds - centroid, axis=1)
                closest_idx = distances.argsort()[:min(3, len(distances))]
                rep_samples = [cluster_texts[i] for i in closest_idx]
            else:
                rep_samples = []

            cluster_info[cluster_id] = {
                "size": cluster_size,
                "keybert_keywords": keybert_kws,
                "tfidf_keywords": tfidf_keywords,
                "representative_samples": rep_samples
            }
        return cluster_info

    # ---------------- cluster naming ----------------
    def generate_cluster_names(self, cluster_info):
        print("Generating cluster names...")
        cluster_names = {}
        for cid, info in cluster_info.items():
            all_keywords = info["keybert_keywords"][:3] + info["tfidf_keywords"][:3]
            unique = []
            seen = set()
            for kw in all_keywords:
                if kw.lower() not in seen:
                    unique.append(kw)
                    seen.add(kw.lower())
            if unique:
                name = " & ".join(unique[:2]).title()
                name = re.sub(r"[^a-zA-Z0-9\s&]", "", name)
            else:
                name = f"Cluster {cid}"
            cluster_names[cid] = name
        return cluster_names

    # ---------------- map to your human categories ----------------
    def map_cluster_labels(self, cluster_names):
        """
        Map numeric cluster ids to your final human-readable categories (cluster_name_mapping).
        Any unmapped cluster -> 'others'. Rows labelled 'other' remain 'others'.
        """
        print("Mapping cluster IDs to descriptive names...")
        # invert cluster_names to id->name already is cluster_names
        id_to_name = {}
        for cid, raw_name in cluster_names.items():
            mapped = self.cluster_name_mapping.get(raw_name, None)
            if mapped is None:
                # try fuzzy match: if any key in cluster_name_mapping is substring of raw_name
                mapped = None
                for k, v in self.cluster_name_mapping.items():
                    if k.lower() in raw_name.lower():
                        mapped = v
                        break
            id_to_name[cid] = mapped if mapped is not None else "others"

        # now apply mapping: numeric ids -> mapped string; 'other' stays 'others'
        def _map_label(x):
            if isinstance(x, (int, np.integer)):
                return id_to_name.get(int(x), "others")
            # sometimes labels are strings "0" from earlier; try int conversion
            try:
                xi = int(x)
                return id_to_name.get(xi, "others")
            except Exception:
                return "others"

        self.df["cluster_label"] = self.df["cluster_label"].apply(_map_label)

    # ---------------- final helpers ----------------
    def visualize_clusters(self):
        print("Creating cluster visualization...")
        # PCA on embeddings for visualization
        pca = PCA(n_components=2, random_state=42)
        embeds_2d = pca.fit_transform(self.embeddings)
        plt.figure(figsize=(12,8))
        # color by original numeric labels before mapping if possible
        color_labels = self.df["cluster_label"].copy()
        # attempt to map string labels to ints for plotting
        try:
            unique = list(pd.Categorical(color_labels).categories)
            codes = pd.Categorical(color_labels).codes
            plt.scatter(embeds_2d[:,0], embeds_2d[:,1], c=codes, cmap="tab10", alpha=0.7)
            plt.colorbar()
        except Exception:
            plt.scatter(embeds_2d[:,0], embeds_2d[:,1], alpha=0.7)
        plt.title(f"Text Clusters Visualization (PCA) — {self.optimal_k} clusters")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.show()

    def print_cluster_summary(self, cluster_info, cluster_names):
        print("\n" + "="*80)
        print("CLUSTER ANALYSIS SUMMARY")
        print("="*80)
        for cid, info in cluster_info.items():
            name = cluster_names.get(cid, f"Cluster {cid}")
            print(f"\nCLUSTER {cid}: {name}")
            print(f"Size: {info['size']}")
            print(f"KeyBERT Keywords: {', '.join(info['keybert_keywords'][:8])}")
            print(f"TF-IDF Keywords: {', '.join(info['tfidf_keywords'][:8])}")
            print("Representative Samples:")
            for i, s in enumerate(info.get("representative_samples", []), 1):
                print(f"  {i}. {s}")
            print("-" * 80)

    # ---------------- main driver (no disk write) ----------------
    def clustering_pipeline(self, visualize=True):
        """
        Run the entire pipeline and return a dataframe (same rows as input)
        with a new column 'cluster_label' (string labels mapped to your categories).
        """
        print("Starting full Text Clustering Pipeline")
        print("="*60)

        # 1) preprocessing
        self.preprocess()

        # 2) load data markers (keeps rows)
        self.load_data()

        # 3) models
        self.initialize_models()

        # 4) embeddings
        self.create_embeddings()

        # 5) find k
        self.find_optimal_clusters()

        # 6) clustering (assign cluster ids; invalid rows labeled 'other')
        self.perform_clustering()

        # 7) extract keywords (KeyBERT + TF-IDF)
        cluster_info = self.extract_cluster_keywords()

        # 8) name clusters
        cluster_names = self.generate_cluster_names(cluster_info)

        # 9) map numeric cluster ids to your final human categories
        self.map_cluster_labels(cluster_names)

        # 10) summary + visualization
        self.print_cluster_summary(cluster_info, cluster_names)
        if visualize:
            self.visualize_clusters()

        # 11) final return: keep all original columns + cluster_label
        print("\nPipeline completed successfully!")
        return self.df.copy()
