import pandas as pd
import sys
import os
import json
import uuid
from datetime import datetime, timedelta
import random
import math

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

# Cerramos la conexión de la parte 1 para evitar conflictos
dm.close()

# ==============================================================================
# BLOQUE 1: SIMULACIÓN DE DATOS (Crear)
# ==============================================================================

# --- CAMBIA ESTO A True PARA ACTIVAR LA SIMULACIÓN ---
EJECUTAR_SIMULACION = True 
# -----------------------------------------------------

if EJECUTAR_SIMULACION:
    # Reabrimos conexión
    dm = DataManagerSimple("database/shopper.db")

    def generate_session(customer_id, start_time, duration_minutes, profile_type="happy"):
        """
        Emociones orgánicas para tu dashboard:
        - JSON con claves: frustration, happiness, surprise, neutral
        - CUST-00020 (frustrated): picos duros de frustración (agujas)
        """
        session_id = str(uuid.uuid4())
        events = []

        step_s = 30
        num_steps = max(2, int((duration_minutes * 60) / step_s))
        current_time = start_time

        def gauss(i, mu, sigma):
            return math.exp(-0.5 * ((i - mu) / sigma) ** 2)

        # ---------- FASES ----------
        if profile_type == "happy":
            phases = [
                ("neutral",   int(num_steps * 0.10)),
                ("happiness", int(num_steps * 0.25)),
                ("neutral",   int(num_steps * 0.10)),
                ("surprise",  int(num_steps * 0.10)),
                ("neutral",   int(num_steps * 0.10)),
                ("happiness", num_steps)
            ]
        else:  # frustrated
            phases = [
                ("neutral",     int(num_steps * 0.10)),
                ("happiness",   int(num_steps * 0.08)),
                ("neutral",     int(num_steps * 0.08)),
                ("frustration", int(num_steps * 0.38)),
                ("neutral",     int(num_steps * 0.08)),
                ("frustration", num_steps)
            ]

        # Ajusta para que sumen exactamente num_steps
        total_ph = sum(d for _, d in phases)
        if total_ph != num_steps:
            phases[-1] = (phases[-1][0], phases[-1][1] + (num_steps - total_ph))

        # ---------- TARGETS ----------
        targets = []
        for phase, d in phases:
            for _ in range(d):
                if phase == "happiness":
                    targets.append({"happiness": 0.92, "frustration": 0.03, "surprise": 0.02})
                elif phase == "frustration":
                    targets.append({"happiness": 0.03, "frustration": 0.92, "surprise": 0.02})
                elif phase == "surprise":
                    targets.append({"happiness": 0.08, "frustration": 0.03, "surprise": 0.90})
                else:
                    targets.append({"happiness": 0.06, "frustration": 0.06, "surprise": 0.03})

        # Estado inicial
        happiness = targets[0]["happiness"]
        frustration = targets[0]["frustration"]
        surprise = targets[0]["surprise"]

        # ---------- SPIKES ----------
        frustration_spike = [0.0] * num_steps
        surprise_spike = [0.0] * num_steps

        if profile_type == "happy":
            # pico suave de sorpresa
            mu = int(num_steps * 0.45)
            sigma = max(1.0, num_steps * 0.03)
            for i in range(num_steps):
                surprise_spike[i] = 0.25 * gauss(i, mu, sigma)
        else:
            # picos DUROS de frustración (agujas)
            spike_positions = [
                int(num_steps * 0.40),
                int(num_steps * 0.47),
                int(num_steps * 0.60),
                int(num_steps * 0.69),
                int(num_steps * 0.78),
            ]
            for pos in spike_positions:
                if 0 <= pos < num_steps:
                    frustration_spike[pos] = random.uniform(0.45, 0.70)
                    if pos + 1 < num_steps:
                        frustration_spike[pos + 1] = random.uniform(0.25, 0.45)
                    if pos + 2 < num_steps:
                        frustration_spike[pos + 2] = random.uniform(0.10, 0.25)

            for pos in spike_positions:
                if 0 <= pos < num_steps:
                    surprise_spike[pos] = random.uniform(0.04, 0.10)

        # ---------- DINÁMICA ----------
        alpha = 0.22  # responde más rápido
        for i in range(num_steps):
            t = targets[i]

            # wobble solo cuando la dominante es alta (para que sea orgánico)
            wobble = 0.02 * math.sin(i * 0.35) + 0.01 * math.sin(i * 0.11)
            th = t["happiness"] + (wobble if t["happiness"] > 0.5 else 0.0)
            tf = t["frustration"] + (wobble if t["frustration"] > 0.5 else 0.0)
            ts = t["surprise"] + (wobble if t["surprise"] > 0.5 else 0.0)

            # añade spikes
            tf += frustration_spike[i]
            ts += surprise_spike[i]

            # límites
            th = max(0.0, min(0.99, th))
            tf = max(0.0, min(0.99, tf))
            ts = max(0.0, min(0.99, ts))

            # EMA hacia target
            happiness = (1 - alpha) * happiness + alpha * th
            frustration = (1 - alpha) * frustration + alpha * tf
            surprise = (1 - alpha) * surprise + alpha * ts

            # ruido mínimo
            happiness += random.uniform(-0.004, 0.004)
            frustration += random.uniform(-0.004, 0.004)
            surprise += random.uniform(-0.004, 0.004)

            # clipping
            happiness = max(0.0, min(0.99, happiness))
            frustration = max(0.0, min(0.99, frustration))
            surprise = max(0.0, min(0.99, surprise))

            # si hay pico de frustración, que domine de verdad
            if profile_type != "happy" and frustration > 0.85:
                happiness *= 0.20
                surprise *= 0.30

            # neutral como resto, pero cap para evitar que se coma todo
            dominant = max(happiness, frustration, surprise)
            neutral_floor = 0.02
            neutral_cap = 0.10 if dominant > 0.80 else 0.30

            neutral = 1.0 - (happiness + frustration + surprise)
            neutral = max(neutral_floor, min(neutral_cap, neutral))

            # normaliza final
            emotions = {
                "happiness": happiness,
                "frustration": frustration,
                "surprise": surprise,
                "neutral": neutral
            }
            total = sum(emotions.values())
            emotions = {k: (v / total) for k, v in emotions.items()}

            dominant_label = max(emotions, key=emotions.get)
            group = dominant_label  # ya coincide con tu dashboard

            intensity = emotions[dominant_label] * 100

            events.append({
                "customer_id": customer_id,
                "session_id": session_id,
                "dominant_label": dominant_label,
                "group_label": group,
                "emotions": json.dumps(emotions),
                "avg_window": round(intensity, 2),
                "timestamp": current_time.strftime('%Y-%m-%d %H:%M:%S')
            })
            current_time += timedelta(seconds=step_s)

        return events


    print("🔄 Generando simulaciones...")
    # Año 2025, Mes 11 (Nov), Día 28, Hora 18:00 (Hora punta) BLACK FRIDAY
    start_time = datetime(2025, 11, 28, 18, 0, 0)

    events_13 = generate_session("CUST-00013", start_time, 45, profile_type="happy")
    events_20 = generate_session("CUST-00020", start_time, 50, profile_type="frustrated")

    all_events = events_13 + events_20
    df_simulated = pd.DataFrame(all_events)

    print(f"✅ Generados {len(df_simulated)} eventos.")
    
    try:
        dm.save_df(df_simulated, "EmotionEvent", if_exists="append")
        print("\n🚀 DATOS GUARDADOS EN LA BASE DE DATOS.")
        print("   Ahora ve a tu Streamlit Manager para ver a CUST-00013 y CUST-00020.")
    except Exception as e:
        print(f"❌ Error guardando en DB: {e}")

    dm.close()
else:
    print("\n⚠️ Simulación desactivada.")


# ==============================================================================
# BLOQUE 2: ELIMINAR LA SIMULACIÓN (Limpiar)
# ==============================================================================

# --- CAMBIA ESTO A True PARA BORRAR LOS DATOS CREADOS ---
ELIMINAR_SIMULACION = False
# --------------------------------------------------------

if ELIMINAR_SIMULACION:
    print("\n🗑️ Iniciando limpieza de simulación...")
    dm = DataManagerSimple("database/shopper.db")

    # 1. Leer tabla actual
    df_all = dm.read_df("EmotionEvent")

    if not df_all.empty:
        # 2. Filtrar: Nos quedamos con todo LO QUE NO SEA CUST-00013 ni CUST-00020
        # El símbolo ~ niega la condición (NOT)
        simulated_ids = ['CUST-00013', 'CUST-00020']
        df_cleaned = df_all[~df_all['customer_id'].isin(simulated_ids)]

        deleted_count = len(df_all) - len(df_cleaned)

        if deleted_count > 0:
            # 3. Sobrescribir la tabla (con if_exists='replace')
            dm.save_df(df_cleaned, "EmotionEvent", if_exists="replace")
            print(f"✅ Se han eliminado {deleted_count} registros de simulación.")
            print("   (CUST-00013 y CUST-00020 han sido borrados de EmotionEvent)")
        else:
            print("⚠️ No se encontraron datos de CUST-00013 ni CUST-00020 para borrar.")
    else:
        print("⚠️ La tabla EmotionEvent ya estaba vacía.")

    dm.close()
else:
    print("⚠️ Limpieza desactivada.")