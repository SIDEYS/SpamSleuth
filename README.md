
# Deceptive Opinion Spam (SpamSleuth)

Our project aims to detect and combat fake or misleading online reviews and comments. Leveraging algorithms of machine learning and Deep Learning, it analyzes language patterns, sentiments, and reviewer behavior to differentiate between genuine and deceptive content. By filtering out deceptive spam, the project enhances consumer trust and ensures more informed decision-making in online marketplaces and review platforms.




## Scope
1)Develop algorithms for detecting deceptive online reviews.

2)Analyze linguistic and behavioral features for spam identification.

3)Ensure scalability for processing large volumes of data.

4)Integrate detection tools into online platforms for practical use.


## Pre-requsites
1)Pandas: A Python library used for data manipulation and analysis, providing data structures and functions to work with structured data efficiently.

2)NumPy: A fundamental package for scientific computing with Python, providing support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions.

3)scikit-learn (sklearn): A machine learning library in Python that provides simple and efficient tools for data mining and data analysis. It includes various algorithms for classification, regression, clustering, and dimensionality reduction.

4)Machine Learning: A field of artificial intelligence that enables systems to learn from data and make predictions or decisions without being explicitly programmed. It encompasses various algorithms and techniques for tasks such as classification, regression, clustering, and reinforcement learning.

5)Deep Learning: A subset of machine learning that focuses on algorithms inspired by the structure and function of the brain's neural networks. Deep learning models, typically implemented with neural networks, are capable of learning representations from large amounts of data for tasks such as image and speech recognition.

6)spaCy: An open-source natural language processing (NLP) library. It provides tools and functionalities for tasks such as tokenization, part-of-speech tagging, named entity recognition, and dependency parsing.

7)GloVe (Global Vectors for Word Representation): GloVe is trained on aggregated global word-word co-occurrence statistics, providing meaningful representations that capture semantic relationships between words.
## Datasets

We used a gold standard dataset which is widely used in the field of deceptive opinion spam.

This corpus contains:

1) 400 truthful positive reviews from TripAdvisor
2) 400 deceptive positive reviews from Mechanical Turk 
3) 400 truthful negative reviews from Expedia, Hotels.com, Orbitz, Priceline, TripAdvisor and Yelp 
4) 400 deceptive negative reviews from Mechanical Turk.
 Each of the above datasets consist of 20 reviews for each of the 20 most Chichago hotels.

link for the dataset:
https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus

or download form the repositry.
## Tech Stack

**Machine Learning:** Data analysis, TfidfVectorizer, CountVectorizer, ML Models, pickle.

**Deep Learning:** Nlp, Spacy, Text Processing, Glove, LSTM's, CNN, Attention models, Exploratory Data Analysis.




## Run Locally

Clone the project

```bash
  git clone https://github.com/SIDEYS/SpamSleuth.git
```

open the notebooks in google collab

```bash
  Upload Dataset in collab
```
```bash
  Upload glove vector files 
```
```bash
  copy the path of the files and paste the link where the files is to be loaded 
```

```bash
  rull all using f9 key or run a single cell
```

