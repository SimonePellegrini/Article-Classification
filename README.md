# News Article Classification

Multiclass classification of news articles into 7 categories, using the article text, title, source and PageRank.
Project by **Simone Pellegrini** (Politecnico di Torino).

## Task

Each article must be assigned to one of seven categories:

| Label | Category |
|-------|----------|
| 0 | International news |
| 1 | Business |
| 2 | Technology |
| 3 | Entertainment |
| 4 | Sports |
| 5 | General news |
| 6 | Health |

- **Development set** (`development.csv`): ~80,000 labeled articles.
- **Evaluation set** (`evaluation.csv`): 20,000 unlabeled articles.
- Fields: `Id`, `source`, `title`, `article` (often raw HTML), `page_rank`, `timestamp`.
- Metric: **macro F1-score**.

## Repository structure

| File | Description |
|------|-------------|
| `main.py` | Final pipeline: trains the model on the development set and writes the predictions for the evaluation set. |
| `final.ipynb` | Notebook with the exploratory analysis, preprocessing, grid searches and error analysis. |
| `finalPyNotebook.pdf` | PDF export of `final.ipynb`, included as a backup in case of execution errors. |
| `ReportPellegriniSimone.pdf` | Full report describing the approach, results and discussion. |
| `development.csv` | Labeled training data. |
| `evaluation.csv` | Unlabeled data to predict. |
| `submission.csv` | Final predictions (`Id`, `Predicted`). |

## Requirements

- Python 3.9+
- `pandas`, `numpy`, `matplotlib`
- `scikit-learn`
- `nltk`
- `beautifulsoup4`

Install everything with:

```bash
pip install pandas numpy matplotlib scikit-learn nltk beautifulsoup4
```

NLTK resources are also needed (tokenizer, lemmatizer, stopwords):

```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("stopwords")
```

## How to run

Place `development.csv` and `evaluation.csv` in the same folder as `main.py`, then run:

```bash
python main.py
```

The script trains the model and writes `submission.csv` in the same folder. The random seed is fixed (`42`), so the results are reproducible.

The exploratory work (plots, grid searches, error analysis) is in `final.ipynb`.

## Approach

### Preprocessing

1. **Balancing**: 30% of the "International news" class is randomly removed (~7,000 samples) to reduce the bias towards the majority class. No oversampling is applied to "Health", which has a very distinctive vocabulary.
2. **HTML cleaning** (BeautifulSoup): tags are stripped, while the `alt`/`src` of images and the text/`href` of links are kept, since they often carry useful keywords (e.g. region, topic). Then HTML entities are unescaped and punctuation is removed.
3. **Text features**: lemmatization (WordNet), English stop-word removal and TF-IDF with n-grams. Title and article use **separate** vectorizers.
4. **Source**: one-hot encoding of the sources that cover 95% of the articles of each class; all the others are grouped into an `other` category.
5. **PageRank**: standardized with `StandardScaler`.
6. `timestamp` and `Id` are discarded (the date distributions are similar across categories).

### Model

A `LinearSVC` (chosen over Random Forest and kernel SVMs for its efficiency on high-dimensional sparse data), with `class_weight="balanced"`.

### Final hyperparameters

| Component | Parameters |
|-----------|-----------|
| LinearSVC | `C=0.1`, `class_weight="balanced"`, `dual=True`, `tol=1e-4` |
| Title TF-IDF | `ngram_range=(1,3)`, `min_df=1`, `max_df=0.3`, `max_features=100000` |
| Article TF-IDF | `ngram_range=(1,5)`, `min_df=1`, `max_df=0.8`, `max_features=100000` |

Hyperparameters were tuned in two independent steps: a randomized search for the TF-IDF parameters (150 candidates, 2-fold CV) and a grid search for the classifier, both with macro F1 as the scoring metric.

## Results

| Evaluation | Macro F1 |
|------------|----------|
| Cross-validation (best random search) | ~0.717 |
| Local hold-out test (20%) | ~0.731 |
| Public leaderboard | ~0.732 |

The similarity between the local and leaderboard scores
