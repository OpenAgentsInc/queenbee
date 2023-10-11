import mysql.connector
from mysql.connector import Error

db_config = {
    "host": "localhost",
    "user": "root",
}

create_stats_container_table = """
CREATE TABLE stats_container (
    id INT AUTO_INCREMENT PRIMARY KEY,
    key VARCHAR(255) NOT NULL
);
"""

create_statsworkers_table = """
CREATE TABLE stats_worker (
    id INT AUTO_INCREMENT PRIMARY KEY,
    stats_container INT NOT NULL,
    bad INT,
    count INT,
    FOREIGN KEY (stats_container) REFERENCES stats_container(id)
);
"""

create_statsbins_table = """
CREATE TABLE stats_bin (
    id INT AUTO_INCREMENT PRIMARY KEY,
    stats_worker_id INT NOT NULL,
    val INT,
    FOREIGN KEY (stats_worker_id) REFERENCES stats_worker(id)
);
"""

def setup_database():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    try:
        print("making the test db")
        cursor.execute("CREATE DATABASE IF NOT EXISTS gputopia_workers")
        cursor.execute("USE gputopia_workers")
        cursor.execute(create_stats_container_table)
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