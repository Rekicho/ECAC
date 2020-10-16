import csv, sqlite3, pandas

from sqlite3 import Error


def create_connection(path):

    connection = None

    try:

        connection = sqlite3.connect(path)

        print("Connection to SQLite DB successful")

    except Error as e:

        print(f"The error '{e}' occurred")

    return connection


def create_schema(path):

    cursor = connection.cursor()

    try: 
        sql_file = open(path)
        cursor.executescript(sql_file.read())
        print("SQLite DB schema created successfully")

    except Error as e:
    
        print(f"The error '{e}' occurred")

def getGender(date):
    if date[2] >= '5':
        return 1
    else:
        return 0

def getBirthday(date):
    if date[2] >= '5':
        date = date[0:2] + str(int(date[2]) - int('5')) + date[3:6]

    return pandas.to_datetime('19' + date, format='%Y%m%d')

def replace_text(df, file):
    if file == "district.csv":
        mapping = {'Prague': 0, 'central Bohemia': 1, 'south Bohemia': 2, 'west Bohemia': 3, 'north Bohemia': 4, 'east Bohemia': 5, 'south Moravia': 6, 'north Moravia': 7}
        df = df.replace({'region': mapping})

    if file == "account.csv":
        mapping = {'monthly issuance': 0, 'issuance after transaction': 1, 'weekly issuance': 2}
        df = df.replace({'frequency': mapping})
        df["date"] = df["date"].apply(lambda x: pandas.to_datetime('19' + x, format='%Y%m%d'))

    if file == "client.csv":
        df["gender"] = df["birth_number"].apply(getGender)
        df["birth_number"] = df["birth_number"].apply(getBirthday)

    if file == "disp.csv":
        mapping = {'OWNER': 0, 'DISPONENT': 1}
        df = df.replace({'type': mapping})

    if file == "trans_train.csv":
        df["date"] = df["date"].apply(lambda x: pandas.to_datetime('19' + x, format='%Y%m%d'))
        mapping = {'credit': 0, 'withdrawal': 1, 'withdrawal in cash': 1}
        df = df.replace({'type': mapping})
        # Maybe also change operation to classes
    
    if file == "trans_test.csv":
        df["date"] = df["date"].apply(lambda x: pandas.to_datetime('19' + x, format='%Y%m%d'))
        mapping = {'credit': 0, 'withdrawal': 1, 'withdrawal in cash': 1}
        df = df.replace({'type': mapping})
        # Maybe also change operation to classes

    if file == "loan_train.csv":
        df["date"] = df["date"].apply(lambda x: pandas.to_datetime('19' + x, format='%Y%m%d'))

    if file == "loan_test.csv":
        df["date"] = df["date"].apply(lambda x: pandas.to_datetime('19' + x, format='%Y%m%d'))

    if file == "card_train.csv":
        df["issued"] = df["issued"].apply(lambda x: pandas.to_datetime('19' + x, format='%Y%m%d'))
        mapping = {'classic': 0, 'junior': 1, 'gold': 2}
        df = df.replace({'type': mapping})

    if file == "card_test.csv":
        df["issued"] = df["issued"].apply(lambda x: pandas.to_datetime('19' + x, format='%Y%m%d'))
        mapping = {'classic': 0, 'junior': 1, 'gold': 2}
        df = df.replace({'type': mapping})
    
    return df
    
        
def populate_table(connection, name, file):
    cur = connection.cursor
    df = pandas.read_csv("data/" + file, sep=';', dtype='unicode')
    df = replace_text(df, file)
    df.to_sql(name, connection, if_exists='append', index=False)
    connection.commit()

def populate(connection):
    populate_table(connection, "District", "district.csv")
    populate_table(connection, "Account", "account.csv")
    populate_table(connection, "Client", "client.csv")
    populate_table(connection, "Disposition", "disp.csv")
    populate_table(connection, "Trans_Train", "trans_train.csv")
    populate_table(connection, "Trans_Test", "trans_test.csv")
    populate_table(connection, "Loan_Train", "loan_train.csv")
    populate_table(connection, "Loan_Test", "loan_test.csv")
    populate_table(connection, "Card_Train", "card_train.csv")
    populate_table(connection, "Card_Test", "card_test.csv")

connection = create_connection("database/bank.db")
# create_schema("database/schema.sql")
# populate(connection)
# connection.cursor().execute("""
#     UPDATE Trans_Train 
#     SET amount = - amount
#     WHERE type == 1;
# """)
# connection.cursor().execute("""
#     UPDATE Trans_Test
#     SET amount = - amount
#     WHERE type == 1;
# """)

def create_dataset():
    train = pandas.read_sql_query("""SELECT loan_id, status, strftime('%m', Loan_Train.date) AS date_month, strftime('%Y', Loan_Train.date) AS date_year, amount, duration, payments,
                            frequency, julianday(Loan_Train.date) - julianday(Account.date) AS account_age, nr_movements, min_amount, avg_amount, max_amount,
                            min_balance, avg_balance, max_balance, sum_amount / ((julianday(max_date) - julianday(min_date)) / 30.0) AS amount_month,
                            CASE WHEN sum_amount / ((julianday(max_date) - julianday(min_date)) / 30.0) > payments THEN 1 ELSE 0 END AS can_pay,
                            credits, withdrawals, (SELECT count(*) FROM Disposition where Account.account_id = Disposition.account_id) AS members,
                            ROUND((julianday(Loan_Train.date) - julianday(Client.birth_number)) / 365.25) AS owner_age,
                            region, inhabitants, municipalities_499, municipalities_1999, municipalities_9999,
                            municipalities_max, cities, ratio_urban_inhabitants, average_salary,
                            unemployment_rate_95, unemployment_rate_96, number_enterpreneurs,
                            (SELECT COUNT(*) FROM Disposition INNER JOIN Card_Train USING(disp_id) WHERE Disposition.account_id = Account.account_id) AS number_credit_cards
                            FROM Loan_Train
                            INNER JOIN Account USING(account_id)
                            INNER JOIN Client ON (SELECT Disposition.client_id FROM Disposition WHERE Disposition.account_id = Account.account_id AND type = 0) = Client.client_id
                            INNER JOIN District ON Client.district_id = District.code
                            LEFT OUTER JOIN
                            (
                                SELECT Trans_Train.account_id, COUNT(*) AS nr_movements,
                                        MIN(amount) AS min_amount, AVG(amount) AS avg_amount, MAX(amount) AS max_amount,
                                        MIN(balance) AS min_balance, AVG(balance) AS avg_balance, MAX(balance) AS max_balance,
                                        MAX(Trans_Train.date) AS max_date, MIN(Trans_Train.date) as min_date, SUM(amount) AS sum_amount,
                                        SUM(CASE WHEN type = 0 THEN 1 ELSE 0 END) as credits,
                                        SUM(CASE WHEN type = 1 THEN 1 ELSE 0 END) as withdrawals
                                FROM Trans_Train
                                GROUP BY Trans_Train.account_id
                            ) AS TT ON Loan_Train.account_id = TT.account_id""", connection)

    test = pandas.read_sql_query("""SELECT loan_id, status, strftime('%m', Loan_Test.date) AS date_month, strftime('%Y', Loan_Test.date) AS date_year, amount, duration, payments,
                            frequency, julianday(Loan_Test.date) - julianday(Account.date) AS account_age, nr_movements, min_amount, avg_amount, max_amount,
                            min_balance, avg_balance, max_balance, sum_amount / ((julianday(max_date) - julianday(min_date)) / 30.0) AS amount_month,
                            CASE WHEN sum_amount / ((julianday(max_date) - julianday(min_date)) / 30.0) > payments THEN 1 ELSE 0 END AS can_pay,
                            credits, withdrawals, (SELECT count(*) FROM Disposition where Account.account_id = Disposition.account_id) AS members,
                            ROUND((julianday(Loan_Test.date) - julianday(Client.birth_number)) / 365.25) AS owner_age,
                            region, inhabitants, municipalities_499, municipalities_1999, municipalities_9999,
                            municipalities_max, cities, ratio_urban_inhabitants, average_salary,
                            unemployment_rate_95, unemployment_rate_96, number_enterpreneurs,
                            (SELECT COUNT(*) FROM Disposition INNER JOIN Card_Test USING(disp_id) WHERE Disposition.account_id = Account.account_id) AS number_credit_cards
                            FROM Loan_Test
                            INNER JOIN Account USING(account_id)
                            INNER JOIN Client ON (SELECT Disposition.client_id FROM Disposition WHERE Disposition.account_id = Account.account_id AND type = 0) = Client.client_id
                            INNER JOIN District ON Client.district_id = District.code
                            LEFT OUTER JOIN
                            (
                                SELECT Trans_Test.account_id, COUNT(*) AS nr_movements,
                                        MIN(amount) AS min_amount, AVG(amount) AS avg_amount, MAX(amount) AS max_amount,
                                        MIN(balance) AS min_balance, AVG(balance) AS avg_balance, MAX(balance) AS max_balance,
                                        MAX(Trans_Test.date) AS max_date, MIN(Trans_Test.date) as min_date, SUM(amount) AS sum_amount,
                                        SUM(CASE WHEN type = 0 THEN 1 ELSE 0 END) as credits,
                                        SUM(CASE WHEN type = 1 THEN 1 ELSE 0 END) as withdrawals
                                FROM Trans_Test
                                GROUP BY Trans_Test.account_id
                            ) AS TT ON Loan_Test.account_id = TT.account_id""", connection)

    print(train.describe())

    return train, test