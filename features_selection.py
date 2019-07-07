
import process_data as DataPrep
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import nltk.corpus

# start with simple bag of words technique
countV = CountVectorizer()
train_count = countV.fit_transform(DataPrep.train_data['text'].values)

def get_countVectorizer_stats():
    train_count.shape
    print(countV.vocabulary_)
    print(countV.get_feature_names()[:25])

print("******** Count Vectorizer Stats: ******** \n")
get_countVectorizer_stats()

# create TF-IDF frequency features
tfidfV = TfidfTransformer()
train_tfidf = tfidfV.fit_transform(train_count)

def get_tfidf_stats():
    train_tfidf.shape
    print(train_tfidf.A[:100])

print("******** TF-IDF Stats: ******** \n")
get_tfidf_stats()

tfidf_ngram = TfidfVectorizer(stop_words='english', ngram_range=(1, 4), use_idf=True, smooth_idf=True)

# POS Tagging
tagged_sentences = nltk.corpus.treebank.tagged_sents()

cutoff = int(.75 * len(tagged_sentences))
training_sentences = DataPrep.train_data['text']

# print("******** Tagged Sentences: ******** \n")
# print(tagged_sentences)
