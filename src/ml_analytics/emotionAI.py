import sys
import os
import pandas as pd
# 1. Obtenemos la ruta absoluta del directorio actual del script
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Subimos dos niveles para llegar a la raíz del proyecto (shopper_git)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
# 3. Añadimos la raíz al path de Python
sys.path.append(project_root)

# AHORA ya puedes importar cosas desde src
from src.db_manager import DataManagerSimple

# Conectamos
dm = DataManagerSimple("database/shopper.db")

# Configuración para que se vea todo
pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.max_colwidth', None) # No recorta el texto de las celdas

try:
    # Leemos la tabla
    df = dm.read_df("EmotionEvent")
    if df.empty:
        print("La tabla está vacía.")
    else:
        print(f"\n--- TOTAL EVENTOS ENCONTRADOS: {len(df)} ---")
        # Mostramos las últimas 5 filas
        print(df.columns)
        print(df.tail(5))
except Exception as e:
    print(f"Error: {e}")