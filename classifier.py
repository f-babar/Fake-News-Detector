import process_data as DataPrep
import features_selection as FeatureSelection

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB


#building classifier using naive bayes
nb_pipeline = Pipeline([
        ('NBCV',FeatureSelection.tfidfV),
        ('nb_clf',MultinomialNB())])

