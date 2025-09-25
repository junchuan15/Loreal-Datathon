# Loreal-Datathon: Solution for Theme 2: CommentSense for Group Lamimi

A full **Python pipeline** for processing YouTube comments to detect spam, check relevance, analyze sentiment, compute actionability, and cluster products. It generates **fact and dimension tables** ready for **Power BI dashboards**.

---

## Features

- **Data Ingestion**: Load and merge YouTube data from a **self-contained staged dataset** (`dataset/staging/data.zip`) containing comments and video information.
- **Spam Detection**: Detect spam using a fine-tuned BERT model.
- **Relevance Filtering**: Keep only comments relevant to the product/brand.
- **Sentiment Analysis**: Compute sentiment polarity and confidence.
- **Actionability Score**: Rank comments by actionability.
- **Product Clustering**: Cluster comments into meaningful product categories using **SentenceTransformers**, **KeyBERT**, and **KMeans**.
- **Product Resonance Score (PRS)**: Combines relevance, actionability, sentiment, and influence to evaluate product engagement.
- **ETL-ready Fact & Dimension Tables**: Export tables for **Power BI dashboards**.

---

## !! Data Preparation

Due to the large size of raw YouTube data, the pipeline expects a **single zipped file** containing all comments and video data:

1. Place your CSV files for comments and videos into `dataset/staging/`.
2. Zip them as `data.zip` in the same folder: `dataset/staging/data.zip`.

---
```bash
## Project Structure
root/
├─ dataset/ # Input and output datasets
│ ├─ staging/ # Raw/staged data (zip file: data.zip)
│ └─ semantic/ # Output fact/dimension CSV tables for Power BI
├─ src/ # Python modules
│ ├─ data_ingestion.py
│ ├─ spam_detector.py
│ ├─ relevance_checker.py
│ ├─ sentiment_analysis.py
│ ├─ compute_actionability.py
│ └─ prod_cat_clustering.py
├─ main.py # Main pipeline driver
└─ requirements.txt # Python dependencies
```
---
## **Getting Started**  

Follow the steps below to set up the project:

### **Step 1: Clone the repository and navigate to the project directory:**  

```bash
git clone [https://github.com/junchuan15/LLM-Open-Ended-Question-Grader-Based-On-Human-Curated-Rubrics.git
cd Loreal-Datathon.git
```

### **Step 2: Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### **Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### **Step 4: Run the pipeline in main.py
```bash
python main.py
```
---

## What Our Solution Does?
- Load YouTube comment data from dataset/staging/data.zip.
- Detect spam and filter irrelevant comments.
- Perform sentiment analysis and compute actionability scores.
- Cluster comments into product categories.
- Compute Product Resonance Scores (PRS).
- Export fact and dimension tables to dataset/semantic/ for Power BI.

## Power BI Data Model
- FactComments: Comment-level metrics including spam, relevance, sentiment, actionability, and PRS.
- DimVideo: Video-level metadata.
- DimAuthor: Author information.
- DimChannel: Channel metadata.
- DimTopicCategory & BridgeVideoTopicCategory: Topic category mappings.
- DimHashtag & BridgeHashtag: Hashtag mappings across videos and comments.

## Dashboard Access
You can view the Power BI dashboard via this link: [Link to Dashboard](https://app.powerbi.com/view?r=eyJrIjoiY2FjYzU4ZWYtYjZmMy00YzY2LWI2YzYtMzAyYWE2YTA5NDAxIiwidCI6ImE2M2JiMWE5LTQ4YzItNDQ4Yi04NjkzLTMzMTdiMDBjYTdmYiIsImMiOjEwfQ%3D%3D)
