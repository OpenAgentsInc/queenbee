from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class StatsContainer(Base):
    __tablename__ = 'stats_container'
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(255), unique=True)
    

class StatsWorker(Base):
    __tablename__ = 'stats_worker'

    id = Column(Integer, primary_key=True, autoincrement=True)
    stats_container_id = Column(Integer, ForeignKey('stats_container.id'))
    bad = Column(Integer)
    count = Column(Integer)
    
    


class StatsBin(Base):
    __tablename__ = 'stats_bin'

    id = Column(Integer, primary_key=True, autoincrement=True)
    stats_worker_id = Column(Integer, ForeignKey('stats_worker.id'))
    msize = Column(Integer)
    val = Column(Integer)


