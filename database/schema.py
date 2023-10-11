from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class StatsContainer(Base):
    __tablename__ = 'stats_container'
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(255), unique=True)
    
    def create(session, name: String):
        existing_stats_container = session.query(StatsContainer).filter_by(key=name).first()
        if existing_stats_container:
            print(f"SatsContainer with key {name} already exists!")
            return existing_stats_container.id, False
        else:
            new_stats_container = StatsContainer(key=name)
            session.add(new_stats_container)
            session.commit()
            print(f"New stats container with key {name} added with ID {new_stats_container.id}")
            return new_stats_container.id, True

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
    val = Column(Integer)


