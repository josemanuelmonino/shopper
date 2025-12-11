from pathlib import Path
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from simulation.sales_simulator import SalesSimulator

# -----------------------------
# Directorios base
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # carpeta SHOPPER
DB_DIR = BASE_DIR / "database"
DATA_DIR = BASE_DIR / "data" / "raw"

# -----------------------------
# 1. Leer CSVs
# -----------------------------
df_clientes = pd.read_csv(DATA_DIR / "df_clientes_final.csv", sep=";", encoding='utf-8')
df_ropa = pd.read_csv(DATA_DIR / "productos_final.csv", sep=";", encoding='utf-8')

# -----------------------------
# 2. Inicializar simulador
# -----------------------------
sim = SalesSimulator(df_clientes, df_ropa)

# -----------------------------
# 3. Ejecutar simulación
# -----------------------------
df_purchase, df_purchase_item, df_items = sim.run_simulation(n_dias=7, items_per_size_category=3)

print(df_purchase)
print(df_purchase_item)
print(df_items)