import mysql.connector
from mysql.connector import Error

db_config = {
    "host": "localhost",
    "user": "root",
}

create_workers_table = """
CREATE TABLE workers (
    key VARCHAR(32) NOT NULL PRIMARY KEY,
    vals TEXT
);
"""


def setup_database():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    try:
        print("making the test db")
        cursor.execute("CREATE DATABASE IF NOT EXISTS gputopia_workers")
        cursor.execute("USE gputopia_workers")
        cursor.execute(create_workers_table)
        connection.commit()
    except Error as e:
        print("Error:", e)
    finally:
        cursor.close()
        connection.close()

def run_tests():
    # tests will go here
    pass

def cleanup_database():
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute("DROP DATABASE IF EXISTS your_database")
        connection.commit()
    except Error as e:
        print("Error:", e)
    finally:
        cursor.close()
        connection.close()

setup_database()
run_tests()
cleanup_database() 