# Sentiment Analysis

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


