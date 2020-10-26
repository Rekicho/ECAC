import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn import tree, svm, preprocessing
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import OneHotEncoder
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from imblearn.over_sampling import SMOTE
import xgboost as xgb

# train_data - Tablee to train model
# data_cols - Collumns to use for training
# status_col - Collumn to predict
# res_data - Table to predict
def classify(train_data, data_cols, status_col, res_data):
    #Replace empty values with NaN
    train_data = train_data.replace('?', np.nan)
    res_data = res_data.replace('?', np.nan)

    train = train_data.to_numpy()

    #Filter data
    x = train[:, data_cols]
    y = train[:, [status_col]].transpose()[0]
    y=y.astype('int')

    imp = SimpleImputer(missing_values=np.nan)
    imp = imp.fit(x)
    x = imp.transform(x)

    oversample = SMOTE()
    x, y = oversample.fit_resample(x, y)

    # x_train, x_test, y_train, y_test = [], [], [], []
    
    # for i in range(0, len(x)):
    #     if x[i][1] == 1996:
    #         x_test.append(x[i])
    #         y_test.append(y[i])
    #     else:
    #         x_train.append(x[i])
    #         y_train.append(y[i])     

    #Split in train and test - Should also try split manually by date
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # oversample = SMOTE()
    # x_train, y_train = oversample.fit_resample(x_train, y_train)

    #Preprocess
    # scaler = preprocessing.StandardScaler().fit(x_train)
    # x_train = scaler.transform(x_train)

    #Train a model
    # classifiers = KNeighborsClassifier()
    # classifier = tree.DecisionTreeClassifier()
    # classifier = svm.LinearSVC()
    classifier = RandomForestClassifier()
    # classifier = xgb.XGBClassifier()
    # classifier = GaussianProcessClassifier()
    # classifier = MLPClassifier(alpha=1, max_iter=1000)
    # classifier = AdaBoostClassifier(cl)
    # classifier = GaussianNB()
    # classifier = QuadraticDiscriminantAnalysis()
    # classifier = LinearDiscriminantAnalysis()
    # classifier = VotingClassifier(
    #     estimators=[('rf', RandomForestClassifier()), ('xgb', xgb.XGBClassifier())],
    #     voting='hard', weights=[2.5,1]
    # )



    # classifier = RandomForestClassifier(300, max_depth=1)
    # best_roc = 0

    # for i in range(0, 10):
    #     i = i+1
    #     for j in range(0, 3):
    #         classifier_i = RandomForestClassifier(300, max_depth=i)
    #         classifier_i = RFECV(classifier_i, scoring='roc_auc')
    #         classifier_i.fit(x_train,y_train)
    #         roc = roc_auc_score(y_test, classifier_i.predict(x_test))
    #         print("I: " + str(i) + " J: " + str(j) + " roc: " + str(roc))
    #         if roc > best_roc:
    #             best_roc = roc
    #             classifier = classifier_i
    #         j = j+1



    Feature Selection
    classifier = RFECV(classifier, scoring='roc_auc')
    
    classifier.fit(x_train,y_train)
    print("Selected Features: %s" % (classifier.ranking_))


    #Test and display stats
    #x_test = scaler.transform(x_test)
    # tree.export_graphviz(classifier, "tree.dot")
    plot_confusion_matrix(classifier, x_test, y_test)
    plt.show()
    
    print("Area under ROC curve: " + str(roc_auc_score(y_test, classifier.predict(x_test))))
    print("Accuracy: " + str(accuracy_score(y_test, classifier.predict(x_test))))
    print("Precision: " + str(precision_score(y_test, classifier.predict(x_test))))
    print("Recall: " + str(recall_score(y_test, classifier.predict(x_test))))
    print("f1: " + str(f1_score(y_test, classifier.predict(x_test))))

    #Predict submission
    res_data["status"] = 0
    res = res_data.to_numpy()

    imp = SimpleImputer(missing_values=np.nan)
    imp = imp.fit(res) 
    res = imp.transform(res)

    x_res = res[:, data_cols]
    y_res = classifier.predict_proba(x_res)
    # y_res = classifier.predict(x_res)
    
    prob_res = []

    for prob in y_res:
        prob_res.append(prob[1]-prob[0])

    #Save to submission file
    res_data['Id'] = res_data['loan_id']
    res_data['Predicted'] = y_res#prob_res
    res_data.to_csv('data/res.csv', index=False, columns=['Id','Predicted'])