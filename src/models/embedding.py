from sqlalchemy import String, ForeignKey, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from src.database.config import Base


class ProgramEmbedding(Base):
    __tablename__ = "program_embeddings"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    program_id: Mapped[int] = mapped_column(
        ForeignKey("programs.id"),
        index=True
    )

    # Tipo narrativo
    section: Mapped[str] = mapped_column(String(100))
    # Ej:
    # "perfil_egresado"
    # "perfil_ocupacional"
    # "diferencial"
    # "requisitos"
    # "descripcion_general"

    # Texto original
    content: Mapped[str] = mapped_column(Text)

    # Para chunking
    chunk_index: Mapped[int] = mapped_column(Integer)

    # Vector embedding
    embedding: Mapped[Vector] = mapped_column(Vector(768))

    program: Mapped["Program"] = relationship(
        back_populates="embeddings"
    )
