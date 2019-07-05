
import process_data as DataPrep
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize

# we will start with simple bag of words technique
# creating feature vector - document term matrix
countV = CountVectorizer()
train_count = countV.fit_transform(DataPrep.train_news['text'].values)

def get_countVectorizer_stats():
    # vocab size
    train_count.shape
    # check vocabulary using below command
    print(countV.vocabulary_)
    # get feature names
    print(countV.get_feature_names()[:25])

print("Count Vectorizer Stats: ******** \n")
get_countVectorizer_stats()

# create tf-df frequency features
# tf-idf
tfidfV = TfidfTransformer()
train_tfidf = tfidfV.fit_transform(train_count)

def get_tfidf_stats():
    train_tfidf.shape
    # get train data feature names
    print(train_tfidf.A[:100])

print("TF-IDF Stats: ******** \n")
get_tfidf_stats()

tfidf_ngram = TfidfVectorizer(stop_words='english', ngram_range=(1, 4), use_idf=True, smooth_idf=True)

# POS Tagging
tagged_sentences = nltk.corpus.treebank.tagged_sents()

cutoff = int(.75 * len(tagged_sentences))
training_sentences = DataPrep.train_news['text']

print("Tagged Sentences: ******** \n")
print(tagged_sentences)
