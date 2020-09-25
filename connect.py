import sqlite3

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
        

connection = create_connection("database/bank.db")
create_schema("database/schema.sql")
