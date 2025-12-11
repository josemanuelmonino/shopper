import random
import pandas as pd

# -----------------------------------------
# 1. Conductas de cliente
# -----------------------------------------
PROFILES = {
    "Minimalista": {
        "p_buy": 0.2,
        "lambda_items": 1.0,
        "price_sensitivity": 0.5,
        "promo_propensity": 0.1,
        "weekend_multiplier": 1.2
    },
    "Cazador de ofertas": {
        "p_buy": 0.8,
        "lambda_items": 2.0,
        "price_sensitivity": 2.0,
        "promo_propensity": 0.7,
        "weekend_multiplier": 1.5
    },
    "Comprador normal": {
        "p_buy": 0.6,
        "lambda_items": 1.5,
        "price_sensitivity": 1.0,
        "promo_propensity": 0.3,
        "weekend_multiplier": 1.3
    },
    "Impulsivo de finde": {
        "p_buy": 0.5,
        "lambda_items": 2.0,
        "price_sensitivity": 1.2,
        "promo_propensity": 0.4,
        "weekend_multiplier": 2.0
    }
}

# Opciones de preferencias
CATEGORIES = ['T-Shirt', 'Sweater', 'Pants', 'Coat']
FABRICS = ['Cotton', 'Wool', 'Polyester']
SEASONS = ['Winter', 'Summer', 'All seasons']
SIZES = ['S','M','L','XL']


def generar_preferencias_ciclicas(index):
    """
    Devuelve preferencias cíclicas basadas en el índice del cliente.
    Así cada cliente recibe preferencias deterministas, no aleatorias.
    """
    return {
        "category": CATEGORIES[index % len(CATEGORIES)],
        "fabric":   FABRICS[index % len(FABRICS)],
        "season":   SEASONS[index % len(SEASONS)],
        "size":     SIZES[index % len(SIZES)]
    }


def generar_perfiles(df_clientes):
    """
    Devuelve un DataFrame con:
    - customer_id
    - gender
    - preferencias (category, fabric, season, size) asignadas cíclicamente
    - perfil_asignado (también cíclico)
    """
    datos = []

    conductas_list = list(PROFILES.keys())

    for i, row in df_clientes.iterrows():
        customer_id = row["customer_id"]
        gender = row["gender"]

        # Perfil cíclico
        perfil = conductas_list[i % len(conductas_list)]

        # Preferencias cíclicas
        prefs = generar_preferencias_ciclicas(i)

        datos.append({
            "customer_id": customer_id,
            "gender": gender,
            "category": prefs["category"],
            "fabric": prefs["fabric"],
            "season": prefs["season"],
            "size": prefs["size"],
            "perfil_asignado": perfil
        })

    return pd.DataFrame(datos), PROFILES