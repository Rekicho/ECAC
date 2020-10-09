import pandas as pd
import matplotlib.pyplot as plt

def analyse_table(table):
    print(table.describe())
    # print("Min: " + str(table["amount"].min()))
    # print("1st Quantile: " + str(table["amount"].quantile(.25)))
    # print("Mean: " + str(table["amount"].mean()))
    # print("3rd Quantile: " + str(table["amount"].quantile(.75)))
    # print("Max: " + str(table["amount"].max()))

    # print("STD: " + str(table["amount"].std()))
    # loan_train.boxplot(column=['amount'])
    # loan_train.boxplot(column=['duration'])
    # loan_train.boxplot(column=['payments'])

    # print("\nStatus:\n" + str(table["status"].value_counts()))

    # print("\nCorrelation:\n" + str(table.corr()))
    # plt.matshow(table.corr())

    table.plot.scatter(x='date', y='amount')
    table.plot.scatter(x='duration', y='amount')
    table.plot.scatter(x='duration', y='payments')
    plt.show()
    

# Para cada tabela, para cada atributo, calcular média, STD.
loan_train = pd.read_csv('data/loan_train.csv', sep=';')

loan_train_93 = loan_train[loan_train["date"].apply(str).str.startswith("93") == 1]
loan_train_94 = loan_train[loan_train["date"].apply(str).str.startswith("94") == 1]
loan_train_95 = loan_train[loan_train["date"].apply(str).str.startswith("95") == 1]
loan_train_96 = loan_train[loan_train["date"].apply(str).str.startswith("96") == 1]

# print("93: " + str(loan_train["date"].apply(str).str.startswith("93").sum()))
# print("94: " + str(loan_train["date"].apply(str).str.startswith("94").sum()))
# print("95: " + str(loan_train["date"].apply(str).str.startswith("95").sum()))
# print("96: " + str(loan_train["date"].apply(str).str.startswith("96").sum()))

print("\n\nloan_train:")
analyse_table(loan_train)
print("\n\nloan_train 93:")
analyse_table(loan_train_93)
print("\n\nloan_train 94:")
analyse_table(loan_train_94)
print("\n\nloan_train 95:")
analyse_table(loan_train_95)
print("\n\nloan_train 96:")
analyse_table(loan_train_96)

loan_test = pd.read_csv('data/loan_test.csv', sep=';')

loan_test_97 = loan_train[loan_train["date"].apply(str).str.startswith("93") == 1]
loan_test_98 = loan_train[loan_train["date"].apply(str).str.startswith("94") == 1]

print("\n\nloan_test:")
analyse_table(loan_test)
print("\n\nloan_test 97:")
analyse_table(loan_test_97)
print("\n\nloan_test 98:")
analyse_table(loan_test_98)