import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('data/loan_train.csv', sep=';')

print("Train data:")
print("Min: " + str(train["amount"].min()))
print("1st Quantile: " + str(train["amount"].quantile(.25)))
print("Mean: " + str(train["amount"].mean()))
print("3rd Quantile: " + str(train["amount"].quantile(.75)))
print("Max: " + str(train["amount"].max()))

print("STD: " + str(train["amount"].std()))

# train.boxplot(column=['amount'])
# train.boxplot(column=['duration'])
# train.boxplot(column=['payments'])

print("\nStatus:\n" + str(train["status"].value_counts()))

print("\nCorrelation:\n" + str(train.corr()))
# plt.matshow(train.corr())
# plt.show()




test = pd.read_csv('data/loan_test.csv', sep=';')

print("\n\nTest data:")
print("Min: " + str(test["amount"].min()))
print("1st Quantile: " + str(test["amount"].quantile(.25)))
print("Mean: " + str(test["amount"].mean()))
print("3rd Quantile: " + str(test["amount"].quantile(.75)))
print("Max: " + str(test["amount"].max()))

print("STD: " + str(test["amount"].std()))

# test.boxplot(column=['amount'])
# plt.show()