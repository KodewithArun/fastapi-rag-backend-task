"""
Database Session and Engine
Sets up a PostgreSQL database connection to store our Document Metadata.
This keeps our actual SQL metadata fully decoupled from our Vector DB payload.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings

# Connected to the PostgreSQL database defined in our configuration
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
