# main.py
import os
import pandas as pd
import numpy as np
from src.data_ingestion import DataIngestionPipeline
from src.spam_detector import SpamDetector
from src.relevance_check import RelevanceChecker
from src.sentiment_analysis import SentimentAnalyzer
from src.compute_actionability import ActionabilityProcessor
from src.prod_cat_clustering import ProductClustering

# ----------------- Setup -----------------
output_folder = "dataset/semantic"
os.makedirs(output_folder, exist_ok=True)

# ----------------- Step 1: Load Data -----------------
ingestion = DataIngestionPipeline("dataset/staging/data.zip")
merged_df = ingestion.ingest_pipeline()

# ----------------- Step 2: Spam Detection -----------------
spam_model = SpamDetector(
    model_name="mrm8488/bert-tiny-finetuned-sms-spam-detection",
    device=-1
)
merged_df = spam_model.spam_detector_pipeline(
    merged_df, text_col="textOriginal", chunk_size=5000
)

# ----------------- Step 3: Relevance Check -----------------
relevance_model = RelevanceChecker(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    device=-1
)
filtered_df = relevance_model.preprocess_comments(
    merged_df, text_col="textOriginal"
)
relevance_df = relevance_model.relevance_check_pipeline(
    filtered_df,
    sem_weight=1.0,
    hashtag_bonus=0.10,
    max_score=1.0,
    threshold=0.4,
    batch_size=64,
    chunk_size=5000
)

# ----------------- Step 4: Sentiment Analysis -----------------
sentiment_model = SentimentAnalyzer()
sentiment_df = sentiment_model.sentiment_pipeline(
    relevance_df, text_column="textOriginal", batch_size=5000
)

# ----------------- Step 5: Actionability -----------------
actionability_model = ActionabilityProcessor()
actionability_df = actionability_model.actionability_pipeline(
    sentiment_df, text_column="textOriginal", batch_size=200000
)

# ----------------- Step 6: Product Clustering -----------------
product_clustering = ProductClustering(actionability_df)
df_final = product_clustering.clustering_pipeline(visualize=False)

# ----------------- Step 7: Calculate Product Resonance Score -----------------
def calculate_product_resonance_score(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    
    # 1. Relevance (R)
    R = df_copy['weighted_relevance']

    # 2. Actionability (A)
    max_actionability = df_copy['actionability_score'].max()
    A = df_copy['actionability_score'] / max_actionability if max_actionability > 0 else 0

    # 3. Sentiment (S)
    conditions = [
        df_copy['sentiment_label'] == 'POSITIVE',
        df_copy['sentiment_label'] == 'NEGATIVE'
    ]
    choices = [1, -1]
    polarity_multiplier = np.select(conditions, choices, default=0)
    polarity_score = polarity_multiplier * df_copy['sentiment_score']
    S = (polarity_score + 1) / 2

    # 4. Influence (I)
    avg_likes_df = df_copy.groupby('videoId').agg(avg_likes=('likeCount_comment', 'mean')).reset_index()
    df_copy = pd.merge(df_copy, avg_likes_df, on='videoId', how='left')
    raw_influence = (df_copy['likeCount_comment'] / df_copy['avg_likes']).fillna(0)
    I = np.clip(raw_influence / 4.0, 0, 1)
    df_copy.drop(columns=['avg_likes'], inplace=True)

    # 5. Final Product Resonance Score (PRS)
    df_copy['product_resonance_score'] = (0.40 * R) + (0.25 * A) + (0.20 * S) + (0.15 * I)

    return df_copy

final_df = calculate_product_resonance_score(df_final)

# ------------------ FACT TABLE ------------------
FactComments = final_df[[
    'commentId', 'videoId', 'authorId',
    'publishedAt_comment', 'likeCount_comment',
    'is_spam', 'spam_score',
    'weighted_relevance', 'is_relevant',
    'cluster_label', 'sentiment_label', 'sentiment_score',
    'actionability_label', 'actionability_score',
    'product_resonance_score'
]]

# ------------------ DIM TABLES ------------------
DimVideo = final_df[[
    'videoId', 'channelId', 'publishedAt_video', 'viewCount',
    'likeCount_video', 'commentCount', 'engagement_rate',
    'contentDuration (seconds)'
]].drop_duplicates('videoId').reset_index(drop=True)

DimAuthor = final_df[['authorId']].drop_duplicates().reset_index(drop=True)
DimChannel = final_df[['channelId']].drop_duplicates().reset_index(drop=True)

# ------------------ TOPIC CATEGORY ------------------
topic_records = []
for idx, row in final_df[['videoId', 'topicCategories']].dropna().iterrows():
    topics = row['topicCategories']
    if isinstance(topics, list):
        for t in topics:
            topic_records.append((row['videoId'], t.strip()))
    elif isinstance(topics, str) and topics.strip():
        for t in [x.strip() for x in topics.replace("|", ",").split(",")]:
            topic_records.append((row['videoId'], t))

df_topic = pd.DataFrame(topic_records, columns=['videoId', 'topicCategory'])
DimTopicCategory = pd.DataFrame({'topicCategory': df_topic['topicCategory'].unique()})
DimTopicCategory['topicCategoryId'] = ['TC_' + str(i) for i in range(1, len(DimTopicCategory) + 1)]
DimTopicCategory = DimTopicCategory[['topicCategoryId', 'topicCategory']]
BridgeVideoTopicCategory = df_topic.merge(DimTopicCategory, on='topicCategory')[['topicCategoryId','videoId']]

# ------------------ HASHTAGS ------------------
video_hashtags = set()
for hv in final_df['hashtags_video'].dropna():
    if isinstance(hv, list):
        video_hashtags.update([h.strip() for h in hv])
    elif isinstance(hv, str) and hv.strip():
        video_hashtags.update([x.strip() for x in hv.replace("|", ",").split(",")])

comment_hashtags = set()
for hc in final_df['hashtags_comments'].dropna():
    if isinstance(hc, list):
        comment_hashtags.update([h.strip() for h in hc])
    elif isinstance(hc, str) and hc.strip():
        comment_hashtags.update([x.strip() for x in hc.replace("|", ",").split(",")])

overlapped_hashtags = video_hashtags.intersection(comment_hashtags)
DimHashtag = pd.DataFrame({'hashtag': sorted(overlapped_hashtags)})
DimHashtag['hashtagId'] = ['HT_' + str(i) for i in range(1, len(DimHashtag) + 1)]

# Build BridgeHashtag
hashtag_records = []
for idx, row in final_df[['videoId', 'hashtags_video']].dropna().iterrows():
    hv = row['hashtags_video']
    tags = hv if isinstance(hv, list) else [x.strip() for x in hv.replace("|", ",").split(",")]
    for h in tags:
        if h in overlapped_hashtags:
            hashtag_records.append((row['videoId'], None, h))

for idx, row in final_df[['commentId', 'hashtags_comments']].dropna().iterrows():
    hc = row['hashtags_comments']
    tags = hc if isinstance(hc, list) else [x.strip() for x in hc.replace("|", ",").split(",")]
    for h in tags:
        if h in overlapped_hashtags:
            hashtag_records.append((None, row['commentId'], h))

BridgeHashtag = pd.DataFrame(hashtag_records, columns=['videoId', 'commentId', 'hashtag'])
BridgeHashtag = BridgeHashtag.merge(DimHashtag, on='hashtag')[['hashtagId','videoId','commentId']]

# ------------------ EXPORT CSV ------------------
FactComments.to_csv(os.path.join(output_folder, "Fact_Comments.csv"), index=False)
DimVideo.to_csv(os.path.join(output_folder, "Dim_Video.csv"), index=False)
DimAuthor.to_csv(os.path.join(output_folder, "Dim_Author.csv"), index=False)
DimChannel.to_csv(os.path.join(output_folder, "Dim_Channel.csv"), index=False)
DimTopicCategory.to_csv(os.path.join(output_folder, "Dim_TopicCategory.csv"), index=False)
BridgeVideoTopicCategory.to_csv(os.path.join(output_folder, "Bridge_Video_TopicCategory.csv"), index=False)
DimHashtag.to_csv(os.path.join(output_folder, "Dim_Hashtag.csv"), index=False)
BridgeHashtag.to_csv(os.path.join(output_folder, "Bridge_Hashtag.csv"), index=False)

print("All tables exported successfully!")
