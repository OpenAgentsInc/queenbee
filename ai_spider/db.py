import os
import mysql.connector



def connect_to_mysql(dictionary: bool):
  MYSQL_HOST = os.environ["MYSQL_HOST"]
  MYSQL_PORT = os.environ["MYSQL_PORT"]
  MYSQL_USER = os.environ["MYSQL_USER"]
  MYSQL_DATABASE = os.environ["MYSQL_DATABASE"]

  db_config = {
    "host": MYSQL_HOST,
    "user": MYSQL_USER,
    "database": MYSQL_DATABASE
  }


  connection = mysql.connector.connect(**db_config)
  if dictionary:
    return connection.cursor(dictionary=True)
  else:  
    return connection.cursor()
