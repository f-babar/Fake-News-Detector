import spacy
import pandas
import nltk
from nltk.stem.porter import *

class FakeNewsDetector:
    '''--------------------------------------
             Removes the Stop Words From text
        --------------------------------------'''

    def removeStopWords(self, doc):
        stopwords = spacy.lang.en.stop_words.STOP_WORDS
        tokens = [token.text for token in doc if not token.is_stop]
        return tokens

    def stemmingWords(self, tokens):
        stemmingTokens = []
        stemmer = PorterStemmer()
        for token in tokens:
            stemmingTokens.append(stemmer.stem(token))
        return stemmingTokens


if __name__ == "__main__":

    # Load English tokenizer, tagger, parser, NER and word vectors
    nlp = spacy.load("en_core_web_sm")

    path = "dataset/Fake.csv"
    data = pandas.read_csv(path)
    fndObj = FakeNewsDetector()

    for i in range(data.shape[0]):
        text = data.iloc[i].values[1]
        # print(text)
        doc = nlp(text)
        # Analyze syntax
        tokens = fndObj.removeStopWords(doc)
        print("Stop Words Removed Tokens:", tokens)
        print("Stemming Tokens:", fndObj.stemmingWords(tokens))
        print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
        print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
        print("**************************************")
        break
