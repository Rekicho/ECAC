import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, roc_auc_score, f1_score
from sklearn import tree, svm, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

DATA_COLS = [4,5] #Duration;Payments
STATUS_COL = 6

#Read data
df = pd.read_csv('data/loan_train.csv', sep=';')
train = df.to_numpy()

#Filter data
x = train[:, DATA_COLS]
y = train[:, [STATUS_COL]].transpose()[0]

#Split in train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

#Preprocess
# scaler = preprocessing.StandardScaler().fit(x_train)
# x_train = scaler.transform(x_train)

#Train a model
# classifier = KNeighborsClassifier(1)
# classifier = tree.DecisionTreeClassifier()
# classifier = svm.LinearSVC()
classifier = RandomForestClassifier()
# classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
classifier.fit(x_train,y_train)
# classifier.fit(x,y)

#Test and display stats
#x_test = scaler.transform(x_test)
plot_confusion_matrix(classifier, x_test, y_test)
plt.show()
print("Area under ROC curve: " + str(roc_auc_score(y_test, classifier.predict(x_test))))
print("f1: " + str(f1_score(y_test, classifier.predict(x_test))))


#Predict submission
df = pd.read_csv('data/loan_test.csv', sep=';')
res = df.to_numpy()
x_res = res[:, DATA_COLS]
y_res = classifier.predict(x_res)

#Save to submission file
df['Id'] = df['loan_id']
df['Predicted'] = y_res
df.to_csv('data/res.csv', index=False, columns=['Id','Predicted'])