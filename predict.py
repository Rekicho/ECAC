import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('data/loan_train.csv', sep=';')
train = df.to_numpy()
x_train = train[:, [4,5]] #duration,payments
y_train = train[:, [6]].transpose()[0]

knn = KNeighborsClassifier()
knn.fit(x_train,y_train)

df = pd.read_csv('data/loan_test.csv', sep=';')
test = df.to_numpy()
x_test = test[:, [4,5]] #duration,payments
y_test = knn.predict(x_test)

df['status'] = y_test
df.to_csv('data/loan_test.csv', index=False, columns=['loan_id','status'])