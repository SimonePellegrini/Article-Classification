import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords as sw
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import re
import html
from nltk.corpus import wordnet as wn
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
import os

class LemmaTokenizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, document):
        lemmas = []
        for t in word_tokenize(document):
            lemma = self.lemmatizer.lemmatize(t)   
            lemmas.append(lemma)

        return lemmas


def compute_top_k_sources(X, y=None,thresh = 0.8):
    sources = set()
    for i in range(7):
        X_i = X[y==i]
        s = X_i["source"].value_counts()
        perc = s.cumsum()/X_i.shape[0]
        s = s[perc<=thresh]
        sources.update(s.keys())
    return sources

def update_sources(X,top_k_sources):
    X["source"] = X["source"].where(X["source"].isin(top_k_sources), other="other")

def clean_html(html_content):
    if not isinstance(html_content, str):
        return ""

    soup = BeautifulSoup(html_content, "html.parser")

    for img in soup.find_all("img"):
        src = img.get("src", "")
        alt = img.get("alt", "")
        replacement = f" {alt} {src} ".strip()
        img.replace_with(replacement)

    for a in soup.find_all("a"):
        href = a.get("href", "")
        text = a.get_text(strip=True)
        replacement = f" {text} {href} ".strip()
        a.replace_with(replacement)
    
    text = soup.get_text(separator=" ")

    text = html.unescape(text)

    text = text.replace("\\", " ")

    text = text.replace("_", " ")

    text = re.sub(r'[()\[\]{}]', ' ', text)

    text = re.sub(r'[^\w\s]', ' ', text)

    return text


def main():

    np.random.seed(42)

    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
   
    # Import development set here!
    df = pd.read_csv("development.csv")

    df = df.dropna()
    
    idx_zero = df[df["label"] == 0].index
    n_drop = int(0.3*len(idx_zero))
    drop_idx = np.random.choice(idx_zero, size=n_drop, replace=False)
    df = df.drop(drop_idx)

    X = df.drop(columns=["Id", "timestamp", "label"], errors='ignore')
    y = df["label"]

    top_k = compute_top_k_sources(X, y, 0.95)
    update_sources(X, top_k)
    
    lemmaTokenizer = LemmaTokenizer()


    preprocessor = ColumnTransformer(
        transformers=[
            ('title', TfidfVectorizer(
                preprocessor=clean_html,
                tokenizer=lemmaTokenizer,
                stop_words="english",
                max_df=0.3,
                min_df=1, 
                ngram_range=(1, 3), 
                max_features=100000
            ), 'title'),
            
            ('article', TfidfVectorizer(
                preprocessor=clean_html,
                tokenizer=lemmaTokenizer,
                stop_words="english",
                max_df=0.8, 
                min_df=1, 
                ngram_range=(1, 5), 
                max_features=100000
            ), 'article'),
            

            ("source", OneHotEncoder(handle_unknown="ignore"), ['source']),
            
            ("PageRank", StandardScaler(), ['page_rank']),
        ],
        remainder='drop' 
    )
    
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearSVC(C=0.1, class_weight="balanced", tol=1e-4,random_state=42))
    ])

    model.fit(X, y)


    # Import evaluation set here!
    X_test = pd.read_csv("evaluation.csv") 
    

    X_test["title"] = X_test["title"].fillna("")
    X_test["article"] = X_test["article"].fillna("")
    
    update_sources(X_test, top_k)
    
    y_pred = model.predict(X_test)
    
    pd.DataFrame({
        "Id": range(len(y_pred)),
        "Predicted": y_pred
    }).to_csv("submission.csv", index=False)
    
    print("Done!")

if __name__ == "__main__":
    main()