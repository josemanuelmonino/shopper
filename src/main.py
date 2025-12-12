from pathlib import Path
import pandas as pd
from setup_db import setup_db  # Tu setup.py
from simulation.sales_simulator import SalesSimulator
from ml_analytics.customer_profile_analytics import CustProfileAnalytics
from ml_analytics.recom_prom_engine import RecommendationEngine
from ml_analytics.dynamic_pricing_engine import DynamicPricingEngine
from db_manager import DataManagerSimple

# Definimos rutas
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
CPA_PATH = MODELS_DIR / "custprofile_model.pkl"
DPE_PATH = MODELS_DIR / "dynamicpricing_model.pkl"
DB_DIR = BASE_DIR / "database"
DB_PATH = DB_DIR / "shopper.db"

def main():
    # ---------------------------------------------------------
    # 1. Inicialización de la base de datos y setup inicial
    # ---------------------------------------------------------
    DB_DIR.mkdir(parents=True, exist_ok=True)

    # Si la base no existe, ejecutamos setup automáticamente
    if not DB_PATH.exists():
        print("❗ No se ha encontrado 'shopper.db'. Ejecutando setup automáticamente...")
        setup_db()
        print("✅ Setup completado.\n")

    # Instanciamos el gestor de datos
    dm = DataManagerSimple()

    # ---------------------------------------------------------
    # 2. Carga de datos desde la base de datos
    # ---------------------------------------------------------
    print("📂 Leyendo datos de la base de datos...")
    df_clientes = dm.read_df("CustomerInfo")
    df_ropa = dm.read_df("Product")
    df_items = dm.read_df("Item")
    print(f"Clientes: {len(df_clientes)}, Productos: {len(df_ropa)}, Items: {len(df_items)}\n")

    # ---------------------------------------------------------
    # 3. Selección de simulación de ventas:
    #    - Si no hay items → simulación inicial
    #    - Si hay items → simulación continuada
    # ---------------------------------------------------------
    if df_items.empty:
        print("🔄 Simulación inicial de ventas (no hay items en base)...")
        salessim = SalesSimulator(df_clientes, df_ropa)
    else:
        print("🔄 Simulación de ventas continuada (ya hay items)...")
        salessim = SalesSimulator(df_clientes, df_ropa, df_items)

    # Preguntamos cuántos días debemos simular
    dias = int(input("⏳ ¿Durante cuántos días quieres simular las ventas? "))
    
    # Ejecutamos la simulación
    df_purchase, df_purchase_item, df_items = salessim.run_simulation(n_dias=dias)

    # Guardamos los resultados
    print("\n📦 Resultados de la simulación de compras")
    print("-"*60)
    dm.save_df(df_purchase, "Purchase", if_exists="append")
    dm.save_df(df_purchase_item, "Purchase_Item", if_exists="append")
    dm.save_df(df_items, "Item", if_exists="append")
    print("✅ Datos de simulación guardados correctamente.\n")

    # ---------------------------------------------------------
    # 4. Analítica de perfiles de clientes (Clustering)
    # ---------------------------------------------------------
    print("="*60)
    print(" INICIALIZANDO ANALÍTICA DE PERFIL DE CLIENTES ")
    print("="*60, "\n")

    # Creamos el analizador de perfiles (KMeans + features)
    CPA = CustProfileAnalytics(df_purchase, df_purchase_item, df_items, df_ropa, model_path=CPA_PATH)
    
    # Guardamos modelo de clustering
    CPA.save_model(path=CPA_PATH)

    # Calculamos el perfil para cada cliente
    df_perfiles = CPA.calculate_customer_profile()
    dm.save_df(df_perfiles, "CustomerProfile")

    # Mostramos distribución del clustering
    print("→ Cantidad de clientes por cluster:")
    print(df_perfiles["cluster"].value_counts().sort_index(), "\n")

    # ---------------------------------------------------------
    # 5. Motor de recomendaciones (basado en similitud y perfiles)
    # ---------------------------------------------------------
    print("→ Creando objeto RecommendationEngine...")
    RCE = RecommendationEngine(df_perfiles, df_purchase_item, CPA.df_clusters, df_ropa, df_items)

    # ---------------------------------------------------------
    # 6. Generación de recomendaciones + promociones por cliente
    # ---------------------------------------------------------
    print("\n🎯 Generando recomendaciones y promociones para cada cliente...\n")
    
    recoms_list = []
    proms_list = []

    # Para cada cliente generamos recomendaciones según su perfil
    for idx, cliente in df_perfiles.iterrows():
        customer_id = cliente["customer_id"]

        # Generamos DataFrames de recomendaciones y promociones
        df_recom, df_prom = RCE.recomendation_promotion_for_customer_df(
            customer_id,
            df_emotions=None,
        )

        recoms_list.append(df_recom)
        proms_list.append(df_prom)

    # Unimos todos los resultados en DataFrames globales
    df_recoms = pd.concat(recoms_list, ignore_index=True)
    df_proms = pd.concat(proms_list, ignore_index=True)

    print("✅ Recomendaciones y promociones generadas para todos los clientes.")
    print(f"→ Total recomendaciones: {len(df_recoms)}")
    print(f"→ Total promociones: {len(df_proms)}")

    # Guardamos en la base de datos
    dm.save_df(df_recoms, "Recommendation")
    dm.save_df(df_proms, "Promotion")
    print("✅ Recomendaciones y promociones guardadas en la base de datos.")

    # ---------------------------------------------------------
    # 7. Motor de Precios Dinámicos
    #    Generamos una simulación especial solo para DP
    # ---------------------------------------------------------
    salessimDP = SalesSimulator(df_clientes, df_ropa)
    
    # Simulamos 180 días con precios actuales (prob_descuento=1.0)
    df_purchase_dp, df_purchase_item_dp, df_items_dp = salessimDP.run_simulation(
        n_dias=180,
        prob_descuento=1.0
    )

    # Creamos el motor de precios dinámicos
    DPE = DynamicPricingEngine(
        df_purchase_dp,
        df_purchase_item_dp,
        df_items_dp,
        df_ropa,
        path=DPE_PATH
    )

    # Calculamos nuevos precios y los guardamos
    df_ropa_dp = DPE.apply_best_prices()
    dm.save_df(df_ropa_dp, "Product", if_exists="replace")

if __name__ == "__main__":
    main()