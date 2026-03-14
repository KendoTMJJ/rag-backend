from typing import List, Optional
from sqlalchemy import String, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.database.config import Base


class Program(Base):
    __tablename__ = "programs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    snies: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True, unique=True)
    program_name: Mapped[str] = mapped_column(String(255), nullable=False)

    division: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    modality: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    duration_semesters: Mapped[Optional[int]
                               ] = mapped_column(Integer, nullable=True)
    total_credits: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True)
    investment_per_semester: Mapped[Optional[str]
                                    ] = mapped_column(String(100), nullable=True)
    year_update: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    start_date: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True)
    schedule: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    degree_awarded: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True)
    qualified_registry: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True)
    link: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    courses: Mapped[List["Course"]] = relationship(
        back_populates="program", cascade="all, delete-orphan")
    degree_options: Mapped[List["DegreeOption"]] = relationship(
        back_populates="program", cascade="all, delete-orphan")
    electives: Mapped[List["Elective"]] = relationship(
        back_populates="program", cascade="all, delete-orphan")
    embeddings: Mapped[List["ProgramEmbedding"]] = relationship(
        back_populates="program", cascade="all, delete-orphan")

    # ✅ NUEVO: narrativa 1:1
    narrative: Mapped[Optional["ProgramNarrative"]] = relationship(
        back_populates="program",
        uselist=False,
        cascade="all, delete-orphan",
    )
