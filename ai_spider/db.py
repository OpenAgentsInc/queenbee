import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base



def connect_to_mysql():
  MYSQL_HOST = os.environ["MYSQL_HOST"]
  MYSQL_PORT = os.environ["MYSQL_PORT"]
  MYSQL_USER = os.environ["MYSQL_USER"]
  MYSQL_DATABASE = os.environ["MYSQL_DATABASE"]

  connection_string = f"mysql+pymysql://{MYSQL_USER}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
  engine = create_engine(connection_string, echo=True)

  Base = declarative_base()
  Base.metadata.create_all(engine)
  SessionLocal = sessionmaker(bind=engine)
  session = SessionLocal()
  return session