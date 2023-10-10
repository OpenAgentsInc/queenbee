import os
import mysql.connector
from mysql.connector import errorcode

MYSQL_HOST = os.environ["MYSQL_HOST"]
MYSQL_PORT = os.environ["MYSQL_PORT"]
MYSQL_USER = os.environ["MYSQL_USER"]
MYSQL_PASSWORD = os.environ["MYSQL_PASSWORD"]
MYSQL_DATABASE = os.environ["MYSQL_DATABASE"]



def connect_to_mysql():
  db_config = {
      "host": MYSQL_HOST,
      "user": MYSQL_USER,
      "database": MYSQL_DATABASE
  }
  try:
    cnx = mysql.connector.connect(**db_config)
  except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
      print("Database access denied")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
      print("Database does not exist")
    else:
      print(err)
  else:
    cnx.close()
  return cnx