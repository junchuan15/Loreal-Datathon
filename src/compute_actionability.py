import re
import pandas as pd
import time
import nltk
from nltk.corpus import stopwords
import emoji

class ActionabilityProcessor:
    def __init__(self):
        # Download stopwords if not already downloaded
        try:
            stopwords.words('english')
        except LookupError:
            nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

        # ----------------- Define Rules -----------------
        self.actionability_labels = [
            "Purchase Intent", "Product Feedback", "Brand Advocacy",
            "Customer Inquiry", "Not Actionable"
        ]

        self.actionability_weights = {
            "Purchase Intent": 3,
            "Product Feedback": 2,
            "Brand Advocacy": 1,
            "Customer Inquiry": 1,
            "Not Actionable": 0
        }

        self.actionability_rules = {
            "Purchase Intent": {
                "keywords": [
                    "buy", "get", "want", "need", "purchase", "shop", "order", "where to buy",
                    "how to get", "price", "cost", "available", "stock", "link", "website",
                    "store", "online", "discount", "sale", "coupon", "promo code", "add to cart",
                    "checkout", "wishlist", "interested in buying", "planning to buy", "gonna buy"
                ],
                "regex_patterns": [
                    r"buy\s+\w+",
                    r"get\s+\w+",
                    r"where\s+to\s+buy",
                    r"how\s+to\s+get",
                    r"price\s+of",
                    r"cost\s+of",
                    r"available\s+in",
                    r"link\s+to",
                    r"website\s+for",
                    r"store\s+location",
                    r"buy\s+it",
                    r"get\s+it",
                    r"want\s+it",
                    r"need\s+it"
                ]
            },
            "Product Feedback": {
                "keywords": [
                    "great", "good", "bad", "poor", "love", "hate", "like", "dislike", "amazing",
                    "terrible", "awesome", "horrible", "ok", "okay", "meh", "works", "doesnt work",
                    "broken", "fixed", "improve", "better", "worse", "favorite", "least favorite",
                    "review", "opinion", "thoughts", "experience", "using", "tried", "smells",
                    "feels", "looks", "packaging", "quality", "value", "texture", "scent", "color",
                    "size", "performance", "durable", "easy to use", "difficult to use", "effective",
                    "ineffective", "worth it", "waste of money", "recommend", "not recommend",
                    "impressed", "disappointed", "happy with", "unhappy with"
                ],
                "regex_patterns": [
                    r"(\w+)\s+is\s+(great|good|bad|poor|amazing|terrible|awesome|horrible|ok|okay|meh)",
                    r"(love|hate|like|dislike)\s+(\w+)",
                    r"(\w+)\s+(works|doesnt work)",
                    r"(\w+)\s+(broken|fixed)",
                    r"improve\s+(\w+)",
                    r"(\w+)\s+is\s+(better|worse)",
                    r"(favorite|least favorite)\s+(\w+)",
                    r"review\s+of\s+(\w+)",
                    r"opinion\s+on\s+(\w+)",
                    r"thoughts\s+on\s+(\w+)",
                    r"experience\s+with\s+(\w+)",
                    r"using\s+(\w+)",
                    r"tried\s+(\w+)",
                    r"(\w+)\s+(smells|feels|looks)",
                    r"(packaging|quality|value|texture|scent|color|size|performance|durable)\s+of\s+(\w+)",
                    r"(easy|difficult)\s+to\s+use",
                    r"(effective|ineffective)\s+(\w+)",
                    r"(worth it|waste of money)",
                    r"(recommend|not recommend)\s+(\w+)",
                    r"(impressed|disappointed)\s+with\s+(\w+)",
                    r"(happy|unhappy)\s+with\s+(\w+)"
                ]
            },
            "Brand Advocacy": {
                "keywords": [
                    "love", "favorite", "amazing", "awesome", "great", "best", "highly recommend",
                    "must have", "game changer", "holy grail", "obsessed", "addicted", "fan",
                    "loyal", "customer", "repurchase", "buy again", "tell everyone", "share",
                    "promote", "support", "stan", "queen", "king", "icon", "legend", "slay",
                    "period", "goals", "inspo", "vibes", "yass", "preach", "blessed", "thank you",
                    "appreciate", "grateful", "excited", "happy", "pleased", "satisfied", "wow",
                    "omg", "ikr", "true", "facts", "relatable", "same", "agreed", "totally",
                    "absolutely", "definitely", "exactly", "nailed it", "on point", "perfect",
                    "flawless", "iconic", "legendary", "masterpiece", "brilliant", "clever",
                    "genius", "innovative", "revolutionary", "pioneer", "trendsetter", "influencer",
                    "inspiration", "role model", "hero", "idol", "goddess", "diva", "superstar",
                    "champion", "winner", "rockstar", "legendary", "epic", "iconic", "classic",
                    "timeless", "unforgettable", "memorable", "impactful", "influential",
                    "transformative", "life-changing", "must-try", "holy grail", "game changer",
                    "worth the hype", "worth every penny", "invest in", "splurge on", "treat yourself",
                    "self-care", "beauty", "skincare", "makeup", "haircare", "fragrance", "fashion",
                    "style", "lifestyle", "wellness", "health", "fitness", "travel", "food",
                    "drink", "music", "movies", "tv shows", "books", "art", "culture", "community",
                    "family", "friends", "love", "peace", "joy", "happiness", "success", "dream",
                    "goal", "passion", "purpose", "motivation", "inspiration", "dedication",
                    "hard work", "perseverance", "resilience", "strength", "courage", "bravery",
                    "kindness", "compassion", "empathy", "gratitude", "mindfulness", "positivity",
                    "optimism", "hope", "faith", "trust", "honesty", "integrity", "respect",
                    "loyalty", "friendship", "relationship", "connection", "community", "support",
                    "teamwork", "collaboration", "unity", "harmony", "balance", "growth", "learning",
                    "development", "progress", "innovation", "creativity", "originality",
                    "uniqueness", "authenticity", "vulnerability", "self-love", "self-acceptance",
                    "self-care", "well-being", "happiness", "joy", "peace", "love"
                ],
                "regex_patterns": [
                    r"highly\s+recommend",
                    r"must\s+have",
                    r"game\s+changer",
                    r"holy\s+grail",
                    r"obsessed\s+with",
                    r"addicted\s+to",
                    r"loyal\s+customer",
                    r"repurchase\s+this",
                    r"buy\s+this\s+again",
                    r"tell\s+everyone\s+about",
                    r"share\s+this\s+with",
                    r"promote\s+(\w+)",
                    r"support\s+(\w+)",
                    r"worth\s+the\s+hype",
                    r"worth\s+every\s+penny",
                    r"invest\s+in\s+(\w+)",
                    r"splurge\s+on\s+(\w+)",
                    r"treat\s+yourself\s+to",
                    r"self-care\s+with",
                    r"(\w+)\s+goals",
                    r"(\w+)\s+inspo",
                    r"(\w+)\s+vibes",
                    r"(\w+)\s+slay",
                    r"(\w+)\s+period",
                    r"thank\s+you\s+for",
                    r"appreciate\s+(\w+)",
                    r"grateful\s+for",
                    r"excited\s+about",
                    r"happy\s+with",
                    r"pleased\s+with",
                    r"satisfied\s+with",
                    r"wow\s+(\w+)",
                    r"omg\s+(\w+)",
                    r"ikr\s+(\w+)",
                    r"true\s+that",
                    r"facts\s+(\w+)",
                    r"relatable\s+(\w+)",
                    r"same\s+here",
                    r"agreed\s+(\w+)",
                    r"totally\s+agree",
                    r"absolutely\s+love",
                    r"definitely\s+recommend",
                    r"exactly\s+what",
                    r"nailed\s+it",
                    r"on\s+point",
                    r"(\w+)\s+is\s+perfect",
                    r"(\w+)\s+is\s+flawless",
                    r"(\w+)\s+is\s+iconic",
                    r"(\w+)\s+is\s+legendary",
                    r"(\w+)\s+is\s+a\s+masterpiece",
                    r"(\w+)\s+is\s+brilliant",
                    r"(\w+)\s+is\s+clever",
                    r"(\w+)\s+is\s+genius",
                    r"(\w+)\s+is\s+innovative",
                    r"(\w+)\s+is\s+revolutionary",
                    r"(\w+)\s+is\s+a\s+pioneer",
                    r"(\w+)\s+is\s+a\s+trendsetter",
                    r"(\w+)\s+is\s+an\s+influencer",
                    r"(\w+)\s+is\s+an\s+inspiration",
                    r"(\w+)\s+is\s+a\s+role\s+model",
                    r"(\w+)\s+is\s+a\s+hero",
                    r"(\w+)\s+is\s+an\s+idol",
                    r"(\w+)\s+is\s+a\s+goddess",
                    r"(\w+)\s+is\s+a\s+diva",
                    r"(\w+)\s+is\s+a\s+superstar",
                    r"(\w+)\s+is\s+a\s+champion",
                    r"(\w+)\s+is\s+a\s+winner",
                    r"(\w+)\s+is\s+a\s+rockstar",
                    r"(\w+)\s+is\s+epic",
                    r"(\w+)\s+is\s+classic",
                    r"(\w+)\s+is\s+timeless",
                    r"(\w+)\s+is\s+unforgettable",
                    r"(\w+)\s+is\s+memorable",
                    r"(\w+)\s+is\s+impactful",
                    r"(\w+)\s+is\s+influential",
                    r"(\w+)\s+is\s+transformative",
                    r"(\w+)\s+is\s+life-changing",
                    r"(\w+)\s+is\s+a\s+must-try",
                    r"@\w+"
                ]
            },
            "Customer Inquiry": {
                "keywords": [
                    "how", "what", "when", "where", "why", "can i", "could i", "is it", "are there",
                    "tell me about", "information on", "question about", "help with", "support for",
                    "contact", "email", "phone", "address", "location", "hours", "website", "app",
                    "account", "order status", "shipping", "return", "refund", "warranty", "manual",
                    "guide", "troubleshooting", "compatible with", "works with", "difference between",
                    "compare to", "alternative to", "best way to", "how long does", "what is the",
                    "when will", "where can i", "why is it", "can you tell me", "could you explain",
                    "is this", "are these", "i have a question", "need help", "looking for information"
                ],
                "regex_patterns": [
                    r"how\s+to\s+\w+",
                    r"what\s+is\s+the\s+\w+",
                    r"when\s+will\s+\w+",
                    r"where\s+can\s+i\s+\w+",
                    r"why\s+is\s+it\s+\w+",
                    r"can\s+i\s+\w+",
                    r"could\s+i\s+\w+",
                    r"is\s+it\s+\w+",
                    r"are\s+there\s+\w+",
                    r"tell\s+me\s+about\s+\w+",
                    r"information\s+on\s+\w+",
                    r"question\s+about\s+\w+",
                    r"help\s+with\s+\w+",
                    r"support\s+for\s+\w+",
                    r"contact\s+(\w+)",
                    r"email\s+address",
                    r"phone\s+number",
                    r"store\s+location",
                    r"opening\s+hours",
                    r"website\s+link",
                    r"mobile\s+app",
                    r"my\s+account",
                    r"order\s+status",
                    r"shipping\s+information",
                    r"return\s+policy",
                    r"get\s+a\s+refund",
                    r"product\s+warranty",
                    r"user\s+manual",
                    r"quick\s+start\s+guide",
                    r"troubleshooting\s+guide",
                    r"compatible\s+with\s+\w+",
                    r"works\s+with\s+\w+",
                    r"difference\s+between\s+\w+\s+and\s+\w+",
                    r"compare\s+\w+\s+to\s+\w+",
                    r"alternative\s+to\s+\w+",
                    r"best\s+way\s+to\s+\w+",
                    r"how\s+long\s+does\s+\w+",
                    r"what\s+is\s+the\s+\w+",
                    r"when\s+will\s+\w+",
                    r"where\s+can\s+i\s+\w+",
                    r"why\s+is\s+it\s+\w+",
                    r"can\s+you\s+tell\s+me",
                    r"could\s+you\s+explain",
                    r"is\s+this\s+\w+",
                    r"are\s+these\s+\w+",
                    r"i\s+have\s+a\s+question",
                    r"need\s+help",
                    r"looking\s+for\s+information"
                ]
            },
            "Not Actionable": {
                "keywords": [
                    "lol", "haha", "🤣", "😂", "😊", "thanks", "thank you", "great video", "nice video",
                    "cool", "awesome", "amazing", "love it", "so true", "ikr", "facts", "relatable",
                    "same", "agreed", "totally", "absolutely", "definitely", "exactly", "nailed it",
                    "on point", "perfect", "flawless", "iconic", "legendary", "masterpiece", "brilliant",
                    "clever", "genius", "innovative", "revolutionary", "pioneer", "trendsetter",
                    "influencer", "inspiration", "role model", "hero", "idol", "goddess", "diva",
                    "superstar", "champion", "winner", "rockstar", "legendary", "epic", "iconic",
                    "classic", "timeless", "unforgettable", "memorable", "impactful", "influential",
                    "transformative", "life-changing", "must-try", "holy grail", "game changer",
                    "worth the hype", "worth every penny", "invest in", "splurge on", "treat yourself",
                    "self-care", "beauty", "skincare", "makeup", "haircare", "fragrance", "fashion",
                    "style", "lifestyle", "wellness", "health", "fitness", "travel", "food",
                    "drink", "music", "movies", "tv shows", "books", "art", "culture", "community",
                    "family", "friends", "love", "peace", "joy", "happiness", "success", "dream",
                    "goal", "passion", "purpose", "motivation", "inspiration", "dedication",
                    "hard work", "perseverance", "resilience", "strength", "courage", "bravery",
                    "kindness", "compassion", "empathy", "gratitude", "mindfulness", "positivity",
                    "optimism", "hope", "faith", "trust", "honesty", "integrity", "respect",
                    "loyalty", "friendship", "relationship", "connection", "community", "support",
                    "teamwork", "collaboration", "unity", "harmony", "balance", "growth", "learning",
                    "development", "progress", "innovation", "creativity", "originality",
                    "uniqueness", "authenticity", "vulnerability",                     "self-love", "self-acceptance",
                    "self-care", "well-being", "happiness", "joy", "peace", "love"
                ]
            }
        }

    # ----------------- Text Cleaning -----------------
    def clean_text(self, text):
        """
        Clean comment text:
        - Remove URLs
        - Convert emojis to text
        - Remove punctuation
        - Lowercase
        - Remove stopwords
        """
        text = str(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = emoji.demojize(text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        text = ' '.join(word for word in text.split() if word not in self.stop_words)
        return text

    # ----------------- Scoring Function -----------------
    def calculate_actionability_score_and_label(self, comment_text):
        """Calculate actionability score and classify label for one comment"""
        score = 0
        category_scores = {category: 0 for category in self.actionability_rules.keys()}

        if pd.isna(comment_text):
            return 0, "Not Actionable"

        for category, category_rules in self.actionability_rules.items():
            weight = self.actionability_weights.get(category, 0)

            for keyword in category_rules.get("keywords", []):
                if keyword.lower() in comment_text.lower():
                    score += weight
                    category_scores[category] += weight

            for pattern in category_rules.get("regex_patterns", []):
                if re.search(pattern, comment_text.lower()):
                    score += weight
                    category_scores[category] += weight

        label = "Not Actionable"
        if any(category_scores.values()):
            label = max(category_scores, key=category_scores.get)

        return score, label

    # ----------------- Pipeline -----------------
    def actionability_pipeline(self, df, text_column="textOriginal", batch_size=200000):
        """
        Full pipeline for actionability scoring in batches.
        Accepts a dataframe, returns the same dataframe with added columns:
          - actionability_score
          - actionability_label
        """
        df = df.copy()
        df['text_cleaned'] = df[text_column].apply(self.clean_text)

        total_rows = len(df)
        results_score = []
        results_label = []

        total_processed = 0
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            batch = df['text_cleaned'].iloc[start:end]

            start_time = time.time()
            batch_results = batch.apply(self.calculate_actionability_score_and_label)
            batch_score = [r[0] for r in batch_results]
            batch_label = [r[1] for r in batch_results]

            results_score.extend(batch_score)
            results_label.extend(batch_label)

            total_processed += len(batch)
            elapsed = time.time() - start_time
            print(f"Processed rows {start} to {end-1} (total {total_processed}) in {elapsed:.2f}s")

        df['actionability_score'] = results_score
        df['actionability_label'] = results_label
        print("Processing complete.")
        return df
