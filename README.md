# Deceptive Opinion Spam Detection (SpamSleuth)

Our project aims to detect and combat **fake or misleading online reviews**. By leveraging **Machine Learning (ML), Deep Learning (DL), and Large Language Models (LLMs)**, it analyzes language patterns, sentiments, and reviewer behavior to differentiate between genuine and deceptive content.  

Filtering out deceptive reviews enhances **consumer trust** and ensures more **informed decision-making** in online marketplaces and review platforms.  

---

## 🚀 Scope
1. Develop algorithms for detecting deceptive online reviews.  
2. Analyze linguistic and behavioral features for spam identification.  
3. Ensure scalability for processing large volumes of data.  
4. Integrate detection tools into online platforms for practical use.  

---

## 📂 Datasets

We used both **benchmark** and **custom datasets**:

- **AMT Dataset (Gold Standard)**  
  - 400 truthful positive reviews from TripAdvisor  
  - 400 deceptive positive reviews from Mechanical Turk  
  - 400 truthful negative reviews from Expedia, Hotels.com, Orbitz, Priceline, TripAdvisor, Yelp  
  - 400 deceptive negative reviews from Mechanical Turk  
  - (20 reviews each for 20 Chicago hotels)  
  - [Kaggle Dataset Link](https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus)  

- **Files Used**  
  - `fake_review_old.csv` → Preprocessed file  
  - `deceptive-opinion.csv` → Raw file  
  - `fake_review.csv` → Custom-generated dataset  
  - `fake_review_dataset_GPT4o.xlsx` → Dataset generated using GPT-4o  

---

## 🛠️ Tech Stack

**Machine Learning**  
- Data Analysis  
- TF-IDF, CountVectorizer  
- scikit-learn models (Logistic Regression, SVM, RandomForest, etc.)  
- pickle  

**Deep Learning**  
- NLP (spaCy, preprocessing, embeddings)  
- GloVe embeddings  
- LSTM, BiLSTM, CNN  
- Attention mechanisms  

**Large Language Models (LLMs)**  
- BERT-based classifiers  
- LLaMA-based detection models  
- GPT-assisted dataset generation  

---

## 📂 Code Organization

### 🔹 Machine Learning
- [basicmlmodelsdecepopspam_newdataset.py](https://github.com/SIDEYS/DeceptiveReviewsLLMs/blob/main/basicmlmodelsdecepopspam_newdataset.py)  
- [basicmlmodelsdecepopspam_olddataset.py](https://github.com/SIDEYS/DeceptiveReviewsLLMs/blob/main/basicmlmodelsdecepopspam_olddataset.py)  

### 🔹 Deep Learning
- [conv1d_bilstm_attention_15_newdataset.py](https://github.com/SIDEYS/DeceptiveReviewsLLMs/blob/main/conv1d_bilstm_attention_15_newdataset.py)  
- [conv1d_lstm_15_epochs_old_dataset.py](https://github.com/SIDEYS/DeceptiveReviewsLLMs/blob/main/conv1d_lstm_15_epochs_old_dataset.py)  
- Variations with different epochs, datasets, and models are similarly named.  

### 🔹 Large Language Models
- [old_fake_reviews+detection_using_bert.py](https://github.com/SIDEYS/DeceptiveReviewsLLMs/blob/main/old_fake_reviews%2Bdetection_using_bert.py)  
- [fake_reviews+detection_using_bert_newdata.py](https://github.com/SIDEYS/DeceptiveReviewsLLMs/blob/main/fake_reviews%2Bdetection_using_bert_newdata.py)  
- [deceptiveopspam_gpt__amtdataset.py](https://github.com/SIDEYS/DeceptiveReviewsLLMs/blob/main/deceptiveopspam_gpt__amtdataset.py)  
- [fake_reviews_detection_using_llama.py](https://github.com/SIDEYS/DeceptiveReviewsLLMs/blob/main/fake_reviews_detection_using_llama.py)  

---

## ⚙️ Prerequisites

- **Python Libraries**  
  - Pandas → data manipulation  
  - NumPy → numerical computing  
  - scikit-learn → ML models  
  - spaCy → NLP preprocessing  
  - TensorFlow / PyTorch → DL models  

- **Embeddings**  
  - GloVe (Global Vectors for Word Representation)  

---

## ▶️ Run Locally

Open notebooks in **Google Colab** or local environment:  

1. Upload dataset files (`fake_review.csv`, `fake_review_old.csv`, etc.)  
2. Upload GloVe vector files  
3. Copy the dataset/vector file paths and update code accordingly  
4. Run cells sequentially (or use **F9** to run all)  

---

## 📧 Contact
For queries, reach out at: **sbhangale@umass.edu**  

---

👉 Repo Link: [SpamSleuth](https://github.com/SIDEYS/SpamSleuth)  
