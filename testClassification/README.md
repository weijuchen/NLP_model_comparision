# NLP Model Comparison

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
