
from sqlalchemy import String, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.database.config import Base

class Course(Base):
    __tablename__ = "courses"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    program_id: Mapped[int] = mapped_column(ForeignKey("programs.id"), index=True)

    semester: Mapped[int] = mapped_column(Integer)
    course_name: Mapped[str] = mapped_column(String(255))
    credits: Mapped[int] = mapped_column(Integer)

    program: Mapped["Program"] = relationship(back_populates="courses")
