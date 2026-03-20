from sqlalchemy import ForeignKey, String, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.database.config import Base

class DegreeOption(Base):
    __tablename__ = "degree_options"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    program_id: Mapped[int] = mapped_column(ForeignKey("programs.id"), index=True)

    option_name: Mapped[str] = mapped_column(String(255))  # "Opción de grado"

    program: Mapped["Program"] = relationship(back_populates="degree_options")
