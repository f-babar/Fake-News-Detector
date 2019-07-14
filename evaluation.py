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
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from warnings import simplefilter
import warnings
warnings.filterwarnings('always')
simplefilter(action='ignore', category=FutureWarning)

print('--------------------------------------------------')
print('K-Fold Cross Validation with Count Vectorization')
print('--------------------------------------------------')

#building classifier using naive bayes
nb_pipeline = Pipeline([
        ('NBCV',FeatureSelection.countV),
        ('nb_clf',MultinomialNB())])

nb_pipeline.fit(DataPrep.train_data['text'],DataPrep.train_data['label'])
predicted_nb = nb_pipeline.predict(DataPrep.test_data['text'])
mean = np.mean(predicted_nb == DataPrep.test_data['label'])

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
        k_fold = KFold(n_splits = 5)
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
                score = f1_score(test_y, predictions, average='weighted', labels=np.unique(predictions))
                scores.append(score)

        return (print('Total texts classified:', len(DataPrep.train_data)),
                print('Score:', sum(scores) / len(scores)),
                print('score length', len(scores)),
                print('Confusion matrix:'),
                print(confusion))

# K-fold cross validation for all classifiers

print('------ NAIVE BAYES ------')
build_confusion_matrix(nb_pipeline)
print('------ Logistic Regression ------')
build_confusion_matrix(logR_pipeline)
print('------ SVM ------')
build_confusion_matrix(svm_pipeline)
print('------ Random Forest ------')
build_confusion_matrix(random_forest)

print('--------------------------------------------------')
print('K-Fold Cross Validation with TF-IDF with N-grams')
print('--------------------------------------------------')

nb_pipeline_ngram = Pipeline([
        ('nb_tfidf',FeatureSelection.tfidf_ngram),
        ('nb_clf',MultinomialNB())])

nb_pipeline_ngram.fit(DataPrep.train_data['text'],DataPrep.train_data['label'])
predicted_nb_ngram = nb_pipeline_ngram.predict(DataPrep.test_data['text'])
np.mean(predicted_nb_ngram == DataPrep.test_data['label'])


#logistic regression classifier
logR_pipeline_ngram = Pipeline([
        ('LogR_tfidf',FeatureSelection.tfidf_ngram),
        ('LogR_clf',LogisticRegression(penalty="l2",C=1, solver='lbfgs'))
        ])

logR_pipeline_ngram.fit(DataPrep.train_data['text'],DataPrep.train_data['label'])
predicted_LogR_ngram = logR_pipeline_ngram.predict(DataPrep.test_data['text'])
np.mean(predicted_LogR_ngram == DataPrep.test_data['label'])


#linear SVM classifier
svm_pipeline_ngram = Pipeline([
        ('svm_tfidf',FeatureSelection.tfidf_ngram),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline_ngram.fit(DataPrep.train_data['text'],DataPrep.train_data['label'])
predicted_svm_ngram = svm_pipeline_ngram.predict(DataPrep.test_data['text'])
np.mean(predicted_svm_ngram == DataPrep.test_data['label'])

# random forest classifier
random_forest_ngram = Pipeline([
        ('rf_tfidf', FeatureSelection.tfidf_ngram),
        ('rf_clf', RandomForestClassifier(n_estimators=300, n_jobs=3))
])

random_forest_ngram.fit(DataPrep.train_data['text'], DataPrep.train_data['label'])
predicted_rf_ngram = random_forest_ngram.predict(DataPrep.test_data['text'])
np.mean(predicted_rf_ngram == DataPrep.test_data['label'])

# K-fold cross validation for all classifiers
print('------ NAIVE BAYES ------')
build_confusion_matrix(nb_pipeline_ngram)
print('------ Logistic Regression ------')
build_confusion_matrix(logR_pipeline_ngram)
print('------ SVM ------')
build_confusion_matrix(svm_pipeline_ngram)
print('------ Random Forest ------')
build_confusion_matrix(random_forest_ngram)

print('--------------------------------------------------')
print('Classification Reports of all classifiers')
print('--------------------------------------------------')

print('------ NAIVE BAYES ------')
print(classification_report(DataPrep.test_data['label'], predicted_nb_ngram))
print('------ Logistic Regression ------')
print(classification_report(DataPrep.test_data['label'], predicted_LogR_ngram))
print('------ SVM ------')
print(classification_report(DataPrep.test_data['label'], predicted_svm_ngram))
print('------ Random Forest ------')
print(classification_report(DataPrep.test_data['label'], predicted_rf_ngram))

# Plotting learing curve
print('--------------------------------------------------')
print('Plot Learning Curves')
print('--------------------------------------------------')


def plot_learing_curve(pipeline, title):
        size = 5
        cv = KFold(size, shuffle=True)

        X = DataPrep.train_data["text"]
        y = DataPrep.train_data["label"]

        pl = pipeline
        pl.fit(X, y)
        train_sizes, train_scores, test_scores = learning_curve(pl, X, y, n_jobs=-1, cv=cv, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.title(title)
        plt.legend(loc="best")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.gca().invert_yaxis()

        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha = 0.1, color="g")

        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

        plt.ylim(-.1, 1.1)
        plt.show()


plot_learing_curve(logR_pipeline_ngram, "Naive-bayes Classifier")
plot_learing_curve(nb_pipeline_ngram, "LogisticRegression Classifier")
plot_learing_curve(svm_pipeline_ngram, "SVM Classifier")
plot_learing_curve(random_forest_ngram, "RandomForest Classifier")
