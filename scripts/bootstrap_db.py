from src.database.init_db import init_db

if __name__ == "__main__":
    try:
        print("Inicializando base de datos...")
        init_db()
        print("Base de datos lista.")
    except Exception as e:
        print(f"Error durante la inicialización: {e}")
        raise
