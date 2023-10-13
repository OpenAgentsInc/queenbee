import mysql.connector
from mysql.connector import Error

db_config = {
    "host": "localhost",
    "user": "root",
}

create_workers_table = """
CREATE TABLE workers (
    identifier VARCHAR(32) NOT NULL PRIMARY KEY,
    vals TEXT
);
"""

def test_setup_database():
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

def test_run_tests():
    # tests will go here
    pass

def test_cleanup_database():
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

test_setup_database()
test_run_tests()
test_cleanup_database() 