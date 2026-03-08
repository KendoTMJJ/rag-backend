from src.core.config import Config
from sqlalchemy import create_engine

from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import DeclarativeBase

# Creamos el motor de la base de datos
engine = create_engine(Config.POSTGRESQL_URL)

# Importante para interactuar con la base de datos, crear sesiones
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close
