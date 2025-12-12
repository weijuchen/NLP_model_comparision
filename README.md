# NLP Model Comparison

This repository demonstrates various NLP approaches for **Sentiment Analysis** and **Text Classification** tasks, comparing traditional machine learning, deep learning, and transformer-based models.

## 📋 Summary

### Sentiment Analysis (Amazon Reviews)
- **VADER**: Rule-based sentiment analyzer
- **DistilBERT**: Transformer-based model (fine-tuned on SST-2)

### Text Classification (20 Newsgroups)
- **Traditional ML**: Logistic Regression (~82.8%), Linear SVC (~85.6%), Naive Bayes
- **Deep Learning**: Text CNN with PyTorch (~80.2%)
- **Transformer**: DistilBERT fine-tuned (~91.3%)

---

# Part I: Sentiment Analysis

Sentiment analysis on Amazon product reviews using traditional and deep learning approaches.

## 📁 File Overview

### `ML_DL_models.ipynb`

Comparison of sentiment analysis methods:

- **VADER (Valence Aware Dictionary and sEntiment Reasoner)**
  - Rule-based sentiment analysis
  - Fast and lightweight
  - No training required
  
- **DistilBERT (Transformer)**
  - Pre-trained model: `distilbert-base-uncased-finetuned-sst-2-english`
  - Fine-tuned on SST-2 dataset
  - High accuracy with deep learning
  - Framework: Hugging Face Transformers

## 📊 Dataset

Amazon product reviews with sentiment labels (positive/negative)

## 🚀 Quick Start

```bash
# Install dependencies
pip install pandas nltk vaderSentiment transformers torch

# Run notebook
jupyter notebook ML_DL_models.ipynb
```

## 🎯 Goal

Compare rule-based (VADER) and transformer-based (DistilBERT) approaches for sentiment analysis tasks.



# Part II: Text Classification

A comparison of different NLP models for text classification on the 20 Newsgroups dataset.

## 📁 Files Overview

### 1. `ML_higherAccuracy.ipynb`

Traditional machine learning approaches for text classification:

- **Models**: Logistic Regression, Linear SVC, Naive Bayes
- **Features**: TF-IDF, Count Vectorizer
- **Dataset**: 20 Newsgroups
- **Accuracy**:
  - Logistic Regression: ~82.8%
  - Linear SVC: ~85.6%

### 2. `DL_model_text_CNN_20newsgroup.ipynb`

Deep learning approach using Convolutional Neural Networks:

- **Model**: Text CNN with PyTorch
- **Dataset**: 20 Newsgroups
- **Features**: Word embeddings
- **Accuracy**: ~80.2%

### 3. `20newsgroups_transformer.ipynb`

State-of-the-art transformer-based approach:

- **Model**: DistilBERT (fine-tuned)
- **Dataset**: 20 Newsgroups
- **Accuracy**: ~91.3%
- **Framework**: Hugging Face Transformers

## 🚀 Quick Start

```bash
# Install dependencies
pip install transformers datasets accelerate scikit-learn torch tqdm

# Run any notebook
jupyter notebook
```

## 📊 Dataset

[20 Newsgroups](https://www.kaggle.com/datasets/crawford/20-newsgroups) - A collection of approximately 20,000 newsgroup documents, partitioned across 20 different newsgroups.

## 🎯 Goal

Compare the performance of traditional ML, deep learning (CNN), and transformer-based models on text classification tasks.

