import pandas as pd
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer

fake_file = 'dataset/Fake.csv'
true_file = 'dataset/True.csv'

fake_news_data = pd.read_csv(fake_file)
true_news_data = pd.read_csv(true_file)
fake_news_data["label"] = 0
true_news_data["label"] = 1

train_data = pd.concat([fake_news_data.head(200),
                        true_news_data.head(200)])

test_data = pd.concat([fake_news_data.tail(200),
                        true_news_data.tail(200)])


def remove_stopwords(data):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    return data.apply(lambda x: ' '
                      .join([word for word in x.split() if word not in (stopwords)]))


def stem_tokens(data):
    stemmer = PorterStemmer()  # SnowballStemmer('english')
    data = data.apply(lambda x: filter(None, x.split(" ")))
    data = data.apply(lambda x : [stemmer.stem(y) for y in x])
    return data.apply(lambda x : " ".join(x))

train_data['text'] = remove_stopwords(train_data['text'])
train_data['text'] = stem_tokens(train_data['text'])

#creating n-grams
def create_unigram(words):
    assert type(words) == list
    return words

#bigram
def create_bigrams(words):
    assert type(words) == list
    skip = 0
    join_str = " "
    Len = len(words)
    if Len > 1:
        lst = []
        for i in range(Len-1):
            for k in range(1,skip+2):
                if i+k < Len:
                    lst.append(join_str.join([words[i],words[i+k]]))
    else:
        #set it as unigram
        lst = create_unigram(words)
    return lst


