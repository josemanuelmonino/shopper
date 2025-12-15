import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from db_manager import DataManagerSimple

# 2. Configuración visual de Pandas (para que no corte las columnas)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 3. Conexión directa
dm = DataManagerSimple("database/shopper.db")

# ==========================================
# PARTE 1: VER EMOTION EVENTS
# ==========================================
print("\n" + "="*50)
print(" 1. ÚLTIMOS EVENTOS DE EMOCIÓN")
print("="*50)

df_emotions = dm.read_df("EmotionEvent")

if not df_emotions.empty:
    # Mostramos las últimas 10 filas
    print(df_emotions.tail(10))
else:
    print("La tabla EmotionEvent está vacía.")

# ==========================================
# PARTE 2: VER CLIENTES CUST-00040 A CUST-00045
# ==========================================
print("\n" + "="*50)
print(" 2. CLIENTES FILTRADOS (40-45)")
print("="*50)

df_customers = dm.read_df("CustomerInfo")

if not df_customers.empty:
    # Filtramos comparando strings directamente
    filtro = (df_customers['customer_id'] >= 'CUST-00040') & \
             (df_customers['customer_id'] <= 'CUST-00045')
    
    subset = df_customers[filtro]
    print(subset)
else:
    print("La tabla CustomerInfo está vacía.")

# 3. FILTRADO
# Opción A: Borrar por ID específico (Más seguro)
# Nos quedamos con los que sean MENORES a CUST-00041 (es decir, hasta el 40 incluido)
# Esto borrará el 42, 43 (duplicado), 44 y 45.
df_limpio = df_customers[df_customers['customer_id'] < 'CUST-00041']

# Opción B (Alternativa): Si quisieras borrar una lista exacta
# ids_a_borrar = ['CUST-00042', 'CUST-00043', 'CUST-00044', 'CUST-00045']
# df_limpio = df[~df['customer_id'].isin(ids_a_borrar)]

print(f"Total después: {len(df_limpio)} filas")
print("\n--- ASÍ QUEDARÁ LA TABLA ---")
print(df_limpio.tail()) # Muestra el final para que confirmes

# 4. GUARDAR (SOBRESCRIBIR)
confirm = input("\n¿Quieres aplicar los cambios en la DB? (s/n): ")
if confirm.lower() == 's':
    # 'replace' borra la tabla vieja y crea una nueva con el df_limpio
    dm.save_df(df_limpio, "CustomerInfo", if_exists="replace")
    print("✅ Tabla actualizada. Usuarios eliminados.")
else:
    print("❌ Operación cancelada.")

dm.close()