import pandas as pd
from datetime import datetime

class RecommendationEngine:
    def __init__(self, df_perfiles, df_ventas_items, df_clusters, df_ropa, df_items):
        """
        df_perfiles: perfiles individuales de cada cliente (cluster y preferencias)
        df_ventas_items: histórico de ventas de items
        df_clusters: información media de cada cluster, incluye top_products
        df_ropa: catálogo actual de ropa
        df_items: items actuales en stock (cada item tiene product_id y talla)
        """
        self.df_perfiles = df_perfiles
        self.df_ventas_items = df_ventas_items
        self.df_clusters = df_clusters
        self.df_ropa = df_ropa
        self.df_items = df_items

        # Combinar items y catálogo
        self.df_catalogo = self.df_items.merge(self.df_ropa, on="product_id", how="left")

        # Calcular top productos por cluster
        self._compute_top_products_per_cluster()

        # Contadores globales para IDs
        self.next_recommendation_id = 1
        self.next_promotion_id = 1

    # ---------------------------------------------------------
    # Calcular top productos más vendidos por cluster
    # ---------------------------------------------------------
    def _compute_top_products_per_cluster(self, top_n=8):
        top_products_dict = {}
        merged = self.df_ventas_items.merge(
            self.df_perfiles[["customer_id", "cluster"]],
            on="customer_id", how="left"
        ).merge(self.df_items, on="item_id", how="left")
        
        for cluster_id in merged["cluster"].unique():
            cluster_sales = merged[merged["cluster"] == cluster_id]
            top_products = cluster_sales["product_id"].value_counts().head(top_n).index.tolist()
            top_products_dict[cluster_id] = top_products

        self.df_clusters["top_products"] = self.df_clusters["cluster"].map(top_products_dict)

    # ---------------------------------------------------------
    # Calcular pesos de productos según preferencias y top products
    # ---------------------------------------------------------
    def _compute_product_weights(self, customer_row, cluster_row, df_emotions=None, df_locations=None):
        df = self.df_catalogo.copy()
        df["weight"] = 1.0

        # Preferencias del cliente
        df.loc[df["category"] == customer_row["category_preference"], "weight"] += 1.3
        df.loc[df["fabric"] == customer_row["fabric_preference"], "weight"] += 1.3

        # Refuerzo para top products del cluster
        top_products = cluster_row.get("top_products", [])
        df.loc[df["product_id"].isin(top_products), "weight"] += 1.6

        # Filtrar items solo de la talla preferida del cliente
        df = df[df["size"] == customer_row["size_preference"]]

        # Normalizar pesos entre 0 y 1
        if not df.empty:
            w = df["weight"]
            df["weight"] = (w - w.min()) / (w.max() - w.min())

        return df

    # ---------------------------------------------------------
    # Recomendar productos y devolver DataFrame listo para DB
    # ---------------------------------------------------------
    def recomendation_promotion_for_customer_df(self, customer_id, df_emotions=None, df_locations=None):
        # Obtener datos del cliente
        row = self.df_perfiles[self.df_perfiles["customer_id"] == customer_id]
        if row.empty:
            raise ValueError(f"Cliente {customer_id} no encontrado.")
        row = row.iloc[0]

        # Obtener cluster
        cluster_id = int(row["cluster"])
        cluster_row = self.df_clusters[self.df_clusters["cluster"] == cluster_id].iloc[0]

        # Calcular pesos
        df_weights = self._compute_product_weights(row, cluster_row,  df_emotions, df_locations)

        # Ordenar por weight descendente
        df_weights = df_weights.sort_values("weight", ascending=False)

        # --- PROMOCIONES (top 3) ---
        top_proms = df_weights.head(3)

        proms_df = pd.DataFrame({
            "promotion_id": range(self.next_promotion_id,
                                self.next_promotion_id + len(top_proms)),
            "customer_id": customer_id,
            "product_id": top_proms["product_id"].values,
            "discount_percentage": ((row["average_discount"] + cluster_row["average_discount"]) / 2)
                                    * top_proms["weight"].values
        })
        self.next_promotion_id += len(top_proms)

        # --- RECOMENDACIONES (los 5 siguientes) ---
        top_recs = df_weights.iloc[3:8]  # posiciones 3,4,5,6,7

        recs_df = pd.DataFrame({
            "recommendation_id": range(self.next_recommendation_id,
                                    self.next_recommendation_id + len(top_recs)),
            "customer_id": customer_id,
            "product_id": top_recs["product_id"].values,
            "weight": top_recs["weight"].values
        })
        self.next_recommendation_id += len(top_recs)

        return recs_df.reset_index(drop=True), proms_df.reset_index(drop=True)