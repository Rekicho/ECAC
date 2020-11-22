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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

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

    # One-hot encoding for region
    enc = OneHotEncoder(handle_unknown='ignore')
    enc_df = pd.DataFrame(enc.fit_transform(train_data[['region']]).toarray())
    train_data = train_data.join(enc_df)

    enc_df = pd.DataFrame(enc.fit_transform(res_data[['region']]).toarray())
    res_data = res_data.join(enc_df)

    train = train_data.to_numpy()

    #Filter data
    x = train[:, data_cols]
    y = train[:, [status_col]].transpose()[0]
    y=y.astype('int')

    imp = SimpleImputer(missing_values=np.nan)
    imp = imp.fit(x)
    x = imp.transform(x)

    # x_train, x_test, y_train, y_test = [], [], [], []
    
    # Split according to year
    # for i in range(0, len(x)):
    #     if x[i][1] == 1996:
    #         x_test.append(x[i])
    #         y_test.append(y[i])
    #     else:
    #         x_train.append(x[i])
    #         y_train.append(y[i])     

    #Split in train and test - Should also try split manually by date
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    oversample = SMOTE()
    x_train, y_train = oversample.fit_resample(x_train, y_train)

    # Use for KNN
    # scaler = preprocessing.StandardScaler().fit(x_train)
    # x_train = scaler.transform(x_train)

    #Train a model
    # classifier = KNeighborsClassifier()
    # classifier = tree.DecisionTreeClassifier()
    # classifier = svm.LinearSVC()
    classifier = RandomForestClassifier(300)
    # classifier = xgb.XGBClassifier()
    # classifier = MLPClassifier(alpha=1, max_iter=1000)
    # classifier = AdaBoostClassifier(cl)
    # classifier = GaussianNB()
    # classifier = VotingClassifier(
    #     estimators=[('dt', tree.DecisionTreeClassifier()), ('svm', svm.LinearSVC()), ('xgb', xgb.XGBClassifier())],
    #     voting='hard', weights=[1,1,1]
    # )
    # classifier = LogisticRegression()


    # Feature Selection
    classifier = RFECV(classifier, scoring='roc_auc')
    
    classifier.fit(x_train,y_train)
    print("Selected Features: %s" % (classifier.ranking_))


    # Test and display stats

    # Use for KNN
    # x_test = scaler.transform(x_test)

    # Save Decision Tree
    # fig = plt.figure(figsize=(50,50))
    # tree.plot_tree(classifier, feature_names=train_data.columns[2:], class_names=['-1', '1'], label='root', filled=True, proportion=True)
    # fig.savefig("decistion_tree.png")

    plot_confusion_matrix(classifier, x_test, y_test)
    plt.show()

    # Print Metrics    
    print("Area under ROC curve: " + str(roc_auc_score(y_test, classifier.predict(x_test))))
    print("Accuracy: " + str(accuracy_score(y_test, classifier.predict(x_test))))
    print("Precision: " + str(precision_score(y_test, classifier.predict(x_test))))
    print("Recall: " + str(recall_score(y_test, classifier.predict(x_test))))
    print("f1: " + str(f1_score(y_test, classifier.predict(x_test))))

    # Predict submission
    res_data["status"] = 0
    res = res_data.to_numpy()

    imp = SimpleImputer(missing_values=np.nan)
    imp = imp.fit(res) 
    res = imp.transform(res)

    x_res = res[:, data_cols]

    # For KNN
    # x_res = scaler.transform(x_res)

    
    y_res = classifier.predict_proba(x_res)

    # Return predicted class only
    # y_res = classifier.predict(x_res)
    
    # Return confidence in prediction
    prob_res = []

    for prob in y_res:
        prob_res.append(prob[1]-prob[0])

    #Save to submission file
    res_data['Id'] = res_data['loan_id']
    res_data['Predicted'] = prob_res
    res_data.to_csv('data/res.csv', index=False, columns=['Id','Predicted'])