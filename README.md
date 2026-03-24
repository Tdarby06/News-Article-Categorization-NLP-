# AG News Article Classification — NLP Project

A multi-class and binary text classification project using the [AG News dataset](https://huggingface.co/datasets/ag_news), comparing traditional ML baselines against deep learning approaches.

---

## Overview

This project classifies news article descriptions into four categories:

| Label | Category |
|-------|----------|
| 0 | World News |
| 1 | Sports |
| 2 | Business |
| 3 | Science / Technology |

For modeling, the task is reframed as **binary classification**: Sports (1) vs. Non-Sports (0).

---

## Dataset

| Split | Articles | Articles per Class |
|-------|----------|--------------------|
| Train | 120,000 | 30,000 |
| Test | 7,600 | 1,900 |

Data is loaded directly via `tensorflow_datasets`:

```python
import tensorflow_datasets as tfds

train_data, test_data = tfds.load(
    'ag_news_subset',
    split=['train', 'test'],
    batch_size=-1
)
```

---

## Setup & Requirements

```bash
pip install tensorflow tensorflow-datasets scikit-learn nltk pandas numpy
pip install matplotlib seaborn wordcloud circlify beautifulsoup4
```

Download required NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')
```

---

## Methodology

### Text Preprocessing
- HTML removal via `BeautifulSoup`
- Punctuation stripping and lowercasing
- Removal of HTML artifact `39s` (misrendered apostrophes)
- Stopword removal via NLTK

### Exploratory Data Analysis
- Word frequency bar charts and circle plots per category
- Word clouds per category
- Sentiment analysis using NLTK VADER (compound scores by label)
- Bigram and trigram analysis per category

**Distinct vocabulary by category:**
- **World News:** government, president, iraq, minister, killed
- **Sports:** coach, cup, season, team, win
- **Business:** million, oil, prices, corp, wall street
- **Science/Tech:** microsoft, software, computer, internet, apple

---

## Models & Results

All models evaluated on **F1 score** (Sports class) on the held-out test set.

| Model | Features | F1 Score |
|-------|----------|----------|
| Logistic Regression | TF-IDF (1-gram) | 0.9512 |
| Logistic Regression | TF-IDF (2-gram) | 0.9517 |
| **Logistic Regression (balanced)** | **TF-IDF (2-gram)** | **0.9555** |
| Random Forest | TF-IDF (1-gram) | 0.9470 |
| Random Forest | TF-IDF (2-gram) | 0.9490 |
| Feedforward Neural Net | Embedding + GAP | 0.9568 |
| 1D CNN | Embedding + Conv1D | 0.9467 |
| Simple RNN | Embedding + RNN | 0.8295 |
| LSTM (multiple variants) | Embedding + LSTM | 0.00 ⚠️ |

> ⚠️ LSTM models failed to converge, predicting only the majority class. Likely caused by a combination of large batch size, class imbalance, and architecture choices.

---

## Recommendation

Despite the feedforward neural network achieving the highest F1 (0.9568), the marginal gain over the **Logistic Regression with balanced class weights** (0.9555) does not justify the added complexity. We recommend the logistic regression model for its:

- Near-identical performance
- Sub-second training time
- Interpretability via model coefficients
- Minimal compute requirements

---

## Key Takeaways

- Traditional ML methods are highly competitive for short-text classification tasks
- N-gram analysis revealed strong category-specific phrase patterns, validating the classification approach
- LSTM models struggled with class imbalance and required further tuning; undersampling experiments did not resolve convergence issues
- Common misclassifications occurred on articles with sports-adjacent content (e.g., financial news about sports organizations, motorsport business coverage)
