from sqlalchemy import ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.database.config import Base


class ProgramNarrative(Base):
    __tablename__ = "program_narratives"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    program_id: Mapped[int] = mapped_column(
        ForeignKey("programs.id"), unique=True, index=True)

    # ✅ Excel nuevo (con significado correcto)
    admission_profile: Mapped[str] = mapped_column(
        Text, nullable=True)        # Perfil del Ingreso
    graduated_profile: Mapped[str] = mapped_column(
        Text, nullable=True)        # Perfil del Egresado
    occupational_profile: Mapped[str] = mapped_column(
        Text, nullable=True)     # Perfil Ocupacional
    differential: Mapped[str] = mapped_column(
        Text, nullable=True)             # Diferencial
    specific_requirements: Mapped[str] = mapped_column(
        Text, nullable=True)    # Requisitos Específicos
    description: Mapped[str] = mapped_column(Text, nullable=True)

    program: Mapped["Program"] = relationship(back_populates="narrative")
