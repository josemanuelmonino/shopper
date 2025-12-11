import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent   # carpeta SHOPPER
DB_DIR = BASE_DIR / "database"
DATA_DIR = BASE_DIR / "data" / "raw"

# Solo se ejecuta una vez
def setup_db():
    # Connect to SQLite
    db_path = DB_DIR / "shopper.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Ejecutar schema.sql
    schema_path = DB_DIR / "schema.sql"
    with open(schema_path, "r", encoding="utf-8") as f:
        cursor.executescript(f.read())

    # Cargar CSVs
    csv_files = [
        (DATA_DIR / "df_clientes_final.csv", "CustomerInfo"),
        (DATA_DIR / "productos_final.csv", "Product"),
    ]

    for csv_path, table_name in csv_files:
        if csv_path.exists():
            df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
            df.to_sql(table_name, conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()