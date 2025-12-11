import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from db_manager import DataManagerSimple

# Supongamos que ya tienes la clase DataManagerSimple definida
dm = DataManagerSimple("database/shopper.db")

# Leer la tabla CustomerProfile en un DataFrame
df_customers = dm.read_df("CustomerInfo")

# Mostrar las primeras filas
print(df_customers.head())

# Cerrar la conexión
dm.close()