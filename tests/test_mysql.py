import mysql.connector
from mysql.connector import Error

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "password",
}

create_workers_table = """
CREATE TABLE Workers (
    worker_id INT AUTO_INCREMENT PRIMARY KEY,
    worker_name VARCHAR(255) NOT NULL
);
"""

create_statsworkers_table = """
CREATE TABLE StatsWorkers (
    stats_worker_id INT AUTO_INCREMENT PRIMARY KEY,
    worker_id INT NOT NULL,
    bad TIME,
    count INT,
    FOREIGN KEY (worker_id) REFERENCES Workers(worker_id)
);
"""

create_statsbins_table = """
CREATE TABLE StatsBins (
    stats_bin_id INT AUTO_INCREMENT PRIMARY KEY,
    stats_worker_id INT NOT NULL,
    val INT,
    FOREIGN KEY (stats_worker_id) REFERENCES StatsWorkers(stats_worker_id)
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
        cursor.execute(create_statsworkers_table)
        cursor.execute(create_statsbins_table)
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