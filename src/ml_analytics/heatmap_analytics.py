import pandas as pd
import numpy as np
import json

RESOLUTION = 20  # resolución de la cuadrícula del heatmap

# --- Función para generar la matriz de densidad ---
def generate_density_matrix(df_trajectories: pd.DataFrame) -> np.ndarray:
    """
    Calcula la matriz de densidad (heatmap crudo) a partir de un DataFrame con columnas 'x' y 'y'.
    """
    min_x, max_x, min_y, max_y = 0, 10, 0, 10
    x_bins = np.linspace(min_x, max_x, RESOLUTION + 1)
    y_bins = np.linspace(min_y, max_y, RESOLUTION + 1)

    df_trajectories['x_bin'] = pd.cut(df_trajectories['x'], bins=x_bins, labels=False, include_lowest=True)
    df_trajectories['y_bin'] = pd.cut(df_trajectories['y'], bins=y_bins, labels=False, include_lowest=True)

    density = df_trajectories.groupby(['x_bin', 'y_bin']).size().reset_index(name='count')

    heatmap_matrix = np.zeros((RESOLUTION, RESOLUTION))
    for _, row in density.iterrows():
        heatmap_matrix[int(row['y_bin']), int(row['x_bin'])] = row['count']

    return heatmap_matrix

# --- Función principal para generar heatmaps ---
def generate_heatmaps_per_customer(df_locations: pd.DataFrame) -> pd.DataFrame:
    """
    Genera heatmaps por customer_id y un heatmap global.
    Devuelve DataFrame con columnas ['customer_id', 'heatmap'].
    """
    records = []

    # Heatmaps por customer_id
    for cust_id, df_cust in df_locations.groupby('customer_id'):
        matrix = generate_density_matrix(df_cust)
        matrix_json = json.dumps(matrix.tolist())  # convierte numpy array a JSON
        records.append({
            "customer_id": cust_id,
            "heatmap": matrix_json
        })

    # Heatmap global
    matrix_global = generate_density_matrix(df_locations)
    matrix_global_json = json.dumps(matrix_global.tolist())
    records.append({
        "customer_id": "CUST_00000",
        "heatmap": matrix_global_json
    })

    return pd.DataFrame(records)