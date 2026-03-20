from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.database.config import Base

class Elective(Base):
    __tablename__ = "electives"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    program_id: Mapped[int] = mapped_column(ForeignKey("programs.id"), index=True)

    course_name: Mapped[str] = mapped_column(String(255))  # Nombre de la Asignatura
    type: Mapped[str] = mapped_column(String(100))         # Tipo

    program: Mapped["Program"] = relationship(back_populates="electives")
