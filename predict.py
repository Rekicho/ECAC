import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn import tree, svm, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE

# train_data - Tablee to train model
# data_cols - Collumns to use for training
# status_cols - Collumn to predict
# res_data - Table to predict
def classify(train_data, data_cols, status_cols, res_data):
    #Replace empty values with NaN
    train_data = train_data.replace('?', np.nan)
    res_data = res_data.replace('?', np.nan)

    train = train_data.to_numpy()
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(train)
    train = imp.transform(train)

    #Filter data
    x = train[:, data_cols]
    y = train[:, [status_cols]].transpose()[0]

    # oversample = SMOTE()
    # x, y = oversample.fit_resample(x, y)

    #Split in train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    #Preprocess
    # scaler = preprocessing.StandardScaler().fit(x_train)
    # x_train = scaler.transform(x_train)

    #Train a model
    # classifier = KNeighborsClassifier()
    # classifier = tree.DecisionTreeClassifier()
    # classifier = svm.LinearSVC(max_iter=10000000)
    classifier = RandomForestClassifier()
    # classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    classifier.fit(x_train,y_train)
    # classifier.fit(x,y)


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

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(res)
    res = imp.transform(res)

    x_res = res[:, data_cols]
    y_res = classifier.predict(x_res)

    #Save to submission file
    res_data['Id'] = res_data['loan_id']
    res_data['Predicted'] = y_res
    res_data.to_csv('data/res.csv', index=False, columns=['Id','Predicted'])