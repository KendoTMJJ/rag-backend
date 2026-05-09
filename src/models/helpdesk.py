
from sqlalchemy import Column, Integer, String, Text, DateTime, LargeBinary, func
from src.database.config import Base


def intent_to_display_label(intent: str) -> str:
    return intent.replace("_", " ").capitalize()


class HelpdeskCategory(Base):
    __tablename__ = "helpdesk_categories"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    intent            = Column(String(100), unique=True, nullable=False)
    document_data     = Column(LargeBinary, nullable=True)
    document_filename = Column(String(255), nullable=True)
    description       = Column(Text, nullable=True)
    updated_at        = Column(DateTime, server_default=func.now(), onupdate=func.now())
