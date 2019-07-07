import process_data as DataPrep
import features_selection as FeatureSelection
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection  import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report


from warnings import simplefilter
import warnings
warnings.filterwarnings('always')
simplefilter(action='ignore', category=FutureWarning)

#building classifier using naive bayes
nb_pipeline = Pipeline([
        ('NBCV',FeatureSelection.countV),
        ('nb_clf',MultinomialNB())])

nb_pipeline.fit(DataPrep.train_data['text'],DataPrep.train_data['label'])
predicted_nb = nb_pipeline.predict(DataPrep.test_data['text'])
np.mean(predicted_nb == DataPrep.test_data['label'])


#building classifier using logistic regression
logR_pipeline = Pipeline([
        ('LogRCV',FeatureSelection.countV),
        ('LogR_clf',LogisticRegression())
])

logR_pipeline.fit(DataPrep.train_data['text'],DataPrep.train_data['label'])
predicted_LogR = logR_pipeline.predict(DataPrep.test_data['text'])
np.mean(predicted_LogR == DataPrep.test_data['label'])

# building Linear SVM classfier
svm_pipeline = Pipeline([
        ('svmCV', FeatureSelection.countV),
        ('svm_clf', svm.LinearSVC())
])

svm_pipeline.fit(DataPrep.train_data['text'], DataPrep.train_data['label'])
predicted_svm = svm_pipeline.predict(DataPrep.test_data['text'])
np.mean(predicted_svm == DataPrep.test_data['label'])

# random forest
random_forest = Pipeline([
        ('rfCV', FeatureSelection.countV),
        ('rf_clf', RandomForestClassifier(n_estimators=200, n_jobs=3))
])

random_forest.fit(DataPrep.train_data['text'], DataPrep.train_data['label'])
predicted_rf = random_forest.predict(DataPrep.test_data['text'])
np.mean(predicted_rf == DataPrep.test_data['label'])


def build_confusion_matrix(classifier):
        k_fold = KFold(n_splits=5)
        scores = []
        confusion = np.array([[0, 0], [0, 0]])

        for train_ind, test_ind in k_fold.split(DataPrep.train_data):
                train_text = DataPrep.train_data.iloc[train_ind]['text']
                train_y = DataPrep.train_data.iloc[train_ind]['label']

                test_text = DataPrep.train_data.iloc[test_ind]['text']
                test_y = DataPrep.train_data.iloc[test_ind]['label']

                classifier.fit(train_text, train_y)
                predictions = classifier.predict(test_text)

                confusion += confusion_matrix(test_y, predictions)
                score = f1_score(test_y, predictions)
                scores.append(score)

        return (print('Total texts classified:', len(DataPrep.train_data)),
                print('Score:', sum(scores) / len(scores)),
                print('score length', len(scores)),
                print('Confusion matrix:'),
                print(confusion))

# K-fold cross validation for all classifiers
build_confusion_matrix(nb_pipeline)
build_confusion_matrix(logR_pipeline)
build_confusion_matrix(svm_pipeline)
build_confusion_matrix(random_forest)
