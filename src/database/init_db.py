from sqlalchemy import text

from src.database.config import engine, Base
import src.models


def init_db() -> None:
    print("--- Inicializando base de datos ---")

    # 1) Asegurar extensión pgvector
    with engine.connect() as conn:
        print("Verificando/activando extensión pgvector...")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
        print("Extensión pgvector lista.")

    # 2) Crear tablas si no existen
    print("Creando tablas...")
    Base.metadata.create_all(bind=engine)
    print("Tablas creadas exitosamente.")

    # 3) Verificación: listar tablas públicas
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT tablename "
                "FROM pg_catalog.pg_tables "
                "WHERE schemaname = 'public';"
            )
        )
        tablas = [row[0] for row in result]
        print(f"Tablas encontradas: {tablas}")
