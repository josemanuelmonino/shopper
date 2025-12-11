import joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

class CustProfileAnalytics:
    def __init__(self, df_purchase, df_purchase_item, df_items, df_ropa, model_path, n_clusters=4):
        self.df_purchase = df_purchase
        self.df_purchase_item = df_purchase_item
        self.df_items = df_items
        self.df_ropa = df_ropa

        # dataset combinado completo
        self.build_full_dataset()

        # inicializamos el scaler y el modelo
        self.scaler = StandardScaler()
        self.model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)

        # comprobar si hay modelo guardado
        if os.path.exists(model_path):
            print("→ Cargando modelo existente y actualizando con partial_fit...")
            self.load_model(model_path)
            self.partial_retrain()
        else:
            print("→ No se encuentra modelo previo. Entrenando desde cero...")
            self.train_initial_model()

    # ------------------------------------------------------------
    # Construcción del dataset completo de compras (join)
    # ------------------------------------------------------------
    def build_full_dataset(self, df_purchase_new=None, df_purchase_item_new=None, df_items_new=None):
        # Concatenar con los datos existentes si se proporcionan nuevos
        df_purchase = pd.concat([self.df_purchase, df_purchase_new]) if df_purchase_new is not None else self.df_purchase
        df_purchase_item = pd.concat([self.df_purchase_item, df_purchase_item_new]) if df_purchase_item_new is not None else self.df_purchase_item
        df_items = pd.concat([self.df_items, df_items_new]) if df_items_new is not None else self.df_items

        # Merge para construir df_full
        df_full = df_purchase_item.merge(
            df_items, on="item_id", how="left"
        ).merge(
            self.df_ropa, on="product_id", how="left"
        ).merge(
            df_purchase, on="purchase_id", how="left", suffixes=("", "_2")
        )

        self.df_full = df_full

    # ------------------------------------------------------------
    # Dataset de características por cliente
    # ------------------------------------------------------------
    def build_customer_feature_matrix(self):
        df = self.df_full

        # Métricas numéricas
        numeric_features = df.groupby("customer_id").agg(
            compras_totales=("purchase_id", "nunique"),
            items_totales=("item_id", "count"),
            gasto_total=("unit_price", "sum"),
            precio_medio=("unit_price", "mean"),
            descuento_medio=("discount", lambda x: x[x != 0].mean() if (x != 0).any() else 0),
            veces_probado=("times_tried", "sum")
        )

        # One-hot de categorías
        df_cat = pd.get_dummies(df[["customer_id", "category"]],
                                columns=["category"], prefix="cat").groupby("customer_id").sum()

        df_fab = pd.get_dummies(df[["customer_id", "fabric"]],
                                columns=["fabric"], prefix="fab").groupby("customer_id").sum()

        df_size = pd.get_dummies(df[["customer_id", "size"]],
                                 columns=["size"], prefix="size").groupby("customer_id").sum()

        # combinación final
        df_features = numeric_features \
            .join(df_cat, how="left") \
            .join(df_fab, how="left") \
            .join(df_size, how="left") \
            .fillna(0)

        return df_features

    # ------------------------------------------------------------
    # Entrenamiento inicial
    # ------------------------------------------------------------
    def train_initial_model(self):
        df_features = self.build_customer_feature_matrix()
        X = self.scaler.fit_transform(df_features)
        self.model.fit(X)

        self.df_clusters = self._centroid_dataframe(df_features)
        return self.df_clusters

    # ------------------------------------------------------------
    # Reentrenamiento incremental
    # ------------------------------------------------------------
    def partial_retrain(self):
        df_features = self.build_customer_feature_matrix()
        X = self.scaler.transform(df_features)
        self.model.partial_fit(X)

        self.df_clusters = self._centroid_dataframe(df_features)
        return self.df_clusters

    # ------------------------------------------------------------
    # Exportar modelo + escalador
    # ------------------------------------------------------------
    def save_model(self, path):
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler
        }, path)

    # ------------------------------------------------------------
    # Importar modelo + escalador
    # ------------------------------------------------------------
    def load_model(self, path):
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]

    # ------------------------------------------------------------
    # Obtener perfil completo de los centroides
    # ------------------------------------------------------------
    def _centroid_dataframe(self, df_features):
        # Desescalar centroides
        centroids = self.scaler.inverse_transform(self.model.cluster_centers_)

        cat_cols = [c for c in df_features.columns if c.startswith("cat_")]
        fab_cols = [c for c in df_features.columns if c.startswith("fab_")]
        size_cols = [c for c in df_features.columns if c.startswith("size_")]

        filas = []
        for cid, centro in enumerate(centroids):
            fila = dict(zip(df_features.columns, centro))

            filas.append({
                "cluster": cid,
                "total_purchases": fila["compras_totales"],
                "total_spent": fila["gasto_total"],
                "average_discount": fila["descuento_medio"],

                "category_preference": max(cat_cols, key=lambda c: fila[c]).replace("cat_", "") if cat_cols else None,
                "fabric_preference":   max(fab_cols, key=lambda c: fila[c]).replace("fab_", "") if fab_cols else None,
                "size_preference":     max(size_cols, key=lambda c: fila[c]).replace("size_", "") if size_cols else None,
            })

        return pd.DataFrame(filas)
    
    # ------------------------------------------------------------
    # Obtener perfil completo de un cliente concreto
    # ------------------------------------------------------------
    def calculate_customer_profile(self, customer_id=None):
        """
        Obtiene el perfil completo de un cliente concreto o de todos los clientes.

        Parámetro opcional:
            customer_id: ID de un cliente específico. Si no se pasa, calcula para todos.

        Retorna:
            dict si se pasa customer_id, DataFrame si se omite.
        """

        # Construir la matriz de características completa
        df_features = self.build_customer_feature_matrix()

        # Función interna para un solo cliente
        def _perfil_cliente(cid):
            if cid not in df_features.index:
                raise ValueError(f"Cliente {cid} sin características válidas para clustering.")

            row = df_features.loc[cid]

            # Cluster
            X_client = row.to_frame().T
            X_scaled = self.scaler.transform(X_client)
            cluster = int(self.model.predict(X_scaled)[0])

            # Métricas numéricas
            compras_totales = row.get("compras_totales", None)
            gasto_total = row.get("gasto_total", None)
            descuento_medio = row.get("descuento_medio", None)

            # Preferencias principales por tipo (one-hot)
            cat_cols = [c for c in df_features.columns if c.startswith("cat_")]
            fab_cols = [c for c in df_features.columns if c.startswith("fab_")]
            size_cols = [c for c in df_features.columns if c.startswith("size_")]

            return {
                "customer_id": cid,
                "cluster": cluster,
                "total_purchases": compras_totales,
                "total_spent": gasto_total,
                "average_discount": descuento_medio,
                "category_preference": row[cat_cols].idxmax().replace("cat_", "") if cat_cols else None,
                "fabric_preference":   row[fab_cols].idxmax().replace("fab_", "") if fab_cols else None,
                "size_preference":     row[size_cols].idxmax().replace("size_", "") if size_cols else None
            }

        # --- Si se pasa un cliente ---
        if customer_id is not None:
            return _perfil_cliente(customer_id)

        # --- Si no se pasa cliente, devolver DataFrame para todos ---
        all_customers = df_features.index
        perfiles = [_perfil_cliente(cid) for cid in all_customers]
        return pd.DataFrame(perfiles)