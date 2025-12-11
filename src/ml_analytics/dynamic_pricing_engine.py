import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime

class DynamicPricingEngine:
    def __init__(self, df_purchase, df_purchase_item, df_items, df_ropa, path):
        self.df_purchase = df_purchase
        self.df_purchase_item = df_purchase_item
        self.df_items = df_items
        self.df_ropa = df_ropa
        self.model_path = path

        # Construir df_daily
        self.df_daily = self._build_daily_sales()

        # Entrenar modelo inicial si no existe
        if not os.path.exists(self.model_path):
            self.initial_train()
        else:
            self.load_model()

    # -------------------------------
    # Construir df_daily agregando ventas y pruebas
    # -------------------------------
    def _build_daily_sales(self):
        """
        Construye un DataFrame diario con ventas agregadas por producto y día.
        Se usa para entrenar el modelo de predicción de demanda.
        """

        # 1. Merge df_purchase_item con df_items para obtener product_id y times_tried
        df = self.df_purchase_item.merge(
            self.df_items[['item_id', 'product_id', 'times_tried']],
            on='item_id',
            how='left'
        )

        # 2. Merge con df_purchase para obtener timestamp de la compra
        df = df.merge(
            self.df_purchase[['purchase_id', 'timestamp']],
            on='purchase_id',
            how='left'
        )

        # 3. Merge con df_ropa para información del producto
        df = df.merge(
            self.df_ropa[['product_id', 'name', 'category', 'gender', 'fabric', 'season',
                        'base_price', 'current_price', 'cost', 'popularity']],
            on='product_id',
            how='left'
        )

        # 4. Convertir timestamp a fecha (solo día)
        df['date'] = pd.to_datetime(df['timestamp']).dt.date

        # 5. Agrupar por fecha y producto
        #    - Contamos product_id → unidades vendidas
        #    - Sumamos times_tried
        #    - Tomamos medias de precios, coste, popularidad
        #    - Tomamos 'first' de las columnas categóricas
        df_daily = df.groupby(['date', 'product_id'], as_index=False).agg({
            'product_id': 'count',       # unidades vendidas
            'current_price': 'mean',        # precio medio de venta
            'cost': 'mean',
            'popularity': 'mean',
            'base_price': 'mean',
            'name': 'first',
            'category': 'first',
            'gender': 'first',
            'fabric': 'first',
            'season': 'first'
        }).rename(columns={'product_id': 'units_sold'})

        return df_daily

    # -------------------------------
    # Preparar dataset LightGBM
    # -------------------------------
    def _get_xy(self, df):
        categorical_features = ['name', 'category', 'gender', 'fabric', 'season']
        features = categorical_features + ['current_price', 'popularity', 'cost', 'base_price']
        target = 'units_sold'

        df_model = df.copy()
        for col in categorical_features:
            df_model[col] = df_model[col].astype('category')

        X = df_model[features]
        y = df_model[target]
        return X, y, categorical_features

    # -------------------------------
    # Entrenamiento inicial
    # -------------------------------
    def initial_train(self):
        X, y, categorical_features = self._get_xy(self.df_daily)
        lgb_train = lgb.Dataset(X, y, categorical_feature=categorical_features)

        params = {
            'objective': 'regression',
            'metric': 'mae',
            'learning_rate': 0.1,
            'num_leaves': 31
        }

        self.model = lgb.train(params, lgb_train, num_boost_round=200)
        self.save_model()

    # -------------------------------
    # Reentrenamiento incremental
    # -------------------------------
    def incremental_train(self):
        X, y, categorical_features = self._get_xy(self.df_daily)
        lgb_train = lgb.Dataset(X, y, categorical_feature=categorical_features)
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'verbose': -1
        }
        self.model = lgb.train(params, lgb_train, num_boost_round=100, init_model=self.model)
        self.save_model()

    # -------------------------------
    # Guardar / cargar modelo
    # -------------------------------
    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"✔ Modelo guardado en: {self.model_path}")

    def load_model(self):
        self.model = joblib.load(self.model_path)
        print("✔ Modelo cargado correctamente")

    # -------------------------------
    # Recomendar precio óptimo para un producto
    # -------------------------------
    def recommend_price_grid(self, row):
        """
        Recorre un grid de precios candidatos para un producto
        y devuelve el precio con la mejor predicción de demanda.
        """
        categorical_features = ['name', 'category', 'gender', 'fabric', 'season']

        best_price = None
        best_profit = -float("inf")

        # Grid de precios = ±20% desde base_price
        price_grid = np.linspace(row["base_price"] * 0.8,
                                row["base_price"] * 1.2,
                                10)

        for p in price_grid:
            feat = {
                "name": row["name"],
                "unit_price": p,
                "popularity": row["popularity"],
                "cost": row["cost"],
                "base_price": row["base_price"],
                "name": row["name"],
                "category": row["category"],
                "gender": row["gender"],
                "fabric": row["fabric"],
                "season": row["season"]
            }

            # Convertir a DataFrame
            feat_df = pd.DataFrame([feat])

            # Aplicar tipos categóricos
            for col in categorical_features:
                feat_df[col] = feat_df[col].astype("category")

            # PREDICCIÓN CORRECTA
            pred = float(self.model.predict(feat_df)[0])
            profit = pred * (p - row["cost"])  # beneficio esperado

            if profit > best_profit:
                best_profit = profit
                best_price = p

        return best_price

    # -------------------------------
    # Actualizar df_ropa con precios recomendados para un día específico
    # -------------------------------
    def apply_best_prices(self):
        updated_df = self.df_ropa.copy()
        for idx, row in updated_df.iterrows():
            best_price = self.recommend_price_grid(row)
            updated_df.at[idx, 'current_price'] = best_price
        return updated_df