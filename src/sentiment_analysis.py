import re
import emoji
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
    
    # ----------------- Text Cleaning -----------------
    @staticmethod
    def clean_text(text: str) -> str:
        text = str(text) if text is not None else ""
        text = emoji.demojize(text, delimiters=(" ", " "))  # Convert emojis to words
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove user mentions
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text
    
    # ----------------- Pipeline -----------------
    def sentiment_pipeline(self, df: pd.DataFrame, text_column: str = "textOriginal", batch_size: int = 5000) -> pd.DataFrame:
        """
        Runs optimized sentiment analysis on a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with a text column.
            text_column (str): Name of the text column (default: "textOriginal").
            batch_size (int): Number of texts per batch.

        Returns:
            pd.DataFrame: DataFrame with sentiment_label and sentiment_score columns.
        """
        # Clean text
        texts = df[text_column].fillna("").astype(str).apply(self.clean_text).tolist()

        all_labels = []
        all_scores = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Tokenize batch
            encoded_input = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            # Model inference
            with torch.no_grad():
                outputs = self.model(**encoded_input)
                scores = torch.softmax(outputs.logits, dim=-1)

            # Decode results
            batch_labels = torch.argmax(scores, dim=1).tolist()
            batch_scores = scores.max(dim=1).values.tolist()

            all_labels.extend([self.model.config.id2label[idx] for idx in batch_labels])
            all_scores.extend(batch_scores)

        # Assign back to DataFrame
        df["sentiment_label"] = all_labels
        df["sentiment_score"] = all_scores

        return df[["sentiment_label", "sentiment_score"]]

