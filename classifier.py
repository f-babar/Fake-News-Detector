import process_data as DataPrep
from sklearn.pipeline import Pipeline
import evaluation as Evaluation
import pickle
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

parameters = {'rf_tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
              'rf_tfidf__use_idf': (True, False),
              'rf_clf__max_depth': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
              }

gs_clf = GridSearchCV(Evaluation.random_forest_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(DataPrep.train_data['text'][:1000], DataPrep.train_data['label'][:1000])

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_

# logistic regression parameters
parameters = {'LogR_tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
              'LogR_tfidf__use_idf': (True, False),
              'LogR_tfidf__smooth_idf': (True, False)
              }

gs_clf = GridSearchCV(Evaluation.logR_pipeline_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(DataPrep.train_data['text'][:1000], DataPrep.train_data['label'][:1000])

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_

# Linear SVM
parameters = {'svm_tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
              'svm_tfidf__use_idf': (True, False),
              'svm_tfidf__smooth_idf': (True, False),
              }

gs_clf = GridSearchCV(Evaluation.svm_pipeline_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(DataPrep.train_data['text'][:1000], DataPrep.train_data['label'][:1000])

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_

random_forest_final = Pipeline([
        ('rf_tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 3), use_idf=True, smooth_idf=True)),
        ('rf_clf', RandomForestClassifier(n_estimators=300, n_jobs=3, max_depth=10))
])

print('--------------------------------------------------')
print('GRID Search parameters optimization')
print('--------------------------------------------------')

random_forest_final.fit(DataPrep.train_data['text'], DataPrep.train_data['label'])
predicted_rf_final = random_forest_final.predict(DataPrep.test_data['text'])
np.mean(predicted_rf_final == DataPrep.test_data['label'])

print('------ Random Forest ------')
print(classification_report(DataPrep.test_data['label'], predicted_rf_final))

logR_pipeline_final = Pipeline([
        # ('LogRCV',countV_ngram),
        ('LogR_tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 5), use_idf=True, smooth_idf=False)),
        ('LogR_clf', LogisticRegression(penalty="l2", C=1, solver='lbfgs'))
])

logR_pipeline_final.fit(DataPrep.train_data['text'], DataPrep.train_data['label'])
predicted_LogR_final = logR_pipeline_final.predict(DataPrep.test_data['text'])
np.mean(predicted_LogR_final == DataPrep.test_data['label'])
# accuracy = 0.62

print('------ Logistic Regression ------')
print(classification_report(DataPrep.test_data['label'], predicted_LogR_final))

#saving best model to the disk
model_file = 'final_model.sav'
pickle.dump(Evaluation.logR_pipeline_ngram, open(model_file, 'wb'))