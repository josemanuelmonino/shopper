import sqlite3
import pandas as pd
from pathlib import Path

class DataManagerSimple:
    def __init__(self, db_path: str = "database/shopper.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)

    def save_df(self, df: pd.DataFrame, table_name: str, if_exists='replace'):
        """
        Guarda un DataFrame en la base de datos.
        
        if_exists: 'replace' | 'append' | 'fail'
        """
        df.to_sql(table_name, self.conn, if_exists=if_exists, index=False)

    def read_df(self, table_name: str) -> pd.DataFrame:
        """Lee una tabla y devuelve un DataFrame."""
        return pd.read_sql(f"SELECT * FROM {table_name}", self.conn)

    def close(self):
        self.conn.close()