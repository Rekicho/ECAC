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
    
        
def populate_table(connection, name, file):
    cur = connection.cursor
    df = pandas.read_csv("data/" + file, sep=';', dtype='unicode')
    df.to_sql(name, connection, if_exists='append', index=False);
    connection.commit()

def populate(connection):
    populate_table(connection, "District", "district.csv")
    populate_table(connection, "Account", "account.csv")
    populate_table(connection, "Client", "client.csv")
    populate_table(connection, "Disposition", "disp.csv")
    populate_table(connection, "Trans", "trans_train.csv")
    populate_table(connection, "Loan", "loan_train.csv")
    populate_table(connection, "Card", "card_train.csv")

connection = create_connection("database/bank.db")
create_schema("database/schema.sql")
populate(connection)
