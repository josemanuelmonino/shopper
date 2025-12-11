import pandas as pd
import random
import numpy as np
import uuid
from datetime import datetime, timedelta
from .artificial_behaviour_generator import generar_perfiles
from .item_generator import ItemGenerator


class SalesSimulator:
    def __init__(self, df_clientes, df_ropa, df_items=None, start_date="2025-01-01"):
        """
        Inicializa el simulador con:
        - df_clientes: DataFrame de clientes (customer_id, gender…)
        - df_ropa: DataFrame con definición de productos (product_id, category, fabric, season, price)
        - start_date: fecha inicial de simulación
        """
        self.df_clientes = df_clientes.copy().reset_index(drop=True)
        self.start_date = pd.to_datetime(start_date)

        # Perfiles de conducta + preferencias
        self.df_perfiles, self.PROFILES = generar_perfiles(self.df_clientes)

        # Generador de ítems (tallas incluidas)
        self.item_generator = ItemGenerator(df_ropa)

        # Guardamos df_ropa como atributo
        self.df_ropa = df_ropa.copy()

        # Datos internos de simulación
        if df_items is not None:
            self.df_items = df_items
        else:
            self.df_items = pd.DataFrame()           # inventario real
        self.df_ventas = pd.DataFrame()              # ventas realizadas (raw)
        self.df_purchase = pd.DataFrame()            # tabla Purchase
        self.df_purchase_item = pd.DataFrame()       # tabla PurchaseItem

    # -------------------------------
    # RESTOCK
    # -------------------------------

    def restock_inicial(self, items_per_size_category=5):
        """Hace un restock inicial desde cero."""
        self.df_items = self.item_generator.restock(
            pd.DataFrame(),
            items_per_size_category=items_per_size_category
        )

    def restock_semanal(self, dia, items_per_size_category=5):
        """Hace restock automáticamente cada 7 días."""
        if dia % 7 == 0:
            self.df_items = self.item_generator.restock(
                self.df_items,
                items_per_size_category=items_per_size_category
            )

    # -------------------------------
    # UTILIDADES
    # -------------------------------

    def _merge_items_with_ropa(self):
        """Fusiona solo el stock disponible con df_ropa para obtener atributos del producto."""
        df_available = self.df_items[self.df_items["status"] == "Available"]

        return df_available.merge(
            self.df_ropa,
            on="product_id",
            how="left"
        )
    
    # -------------------------------
    # CÁLCULO DE PESOS
    # -------------------------------

    def _compute_product_weights(self, cliente):
        df_ropa = self.df_ropa.copy()
        perfil = cliente["perfil_asignado"]

        # Preferencias base
        w_cat = (df_ropa["category"] == cliente["category"]).astype(float) * 1.5
        w_fab = (df_ropa["fabric"] == cliente["fabric"]).astype(float) * 1.3
        w_sea = (df_ropa["season"] == cliente["season"]).astype(float) * 1.2
        base_pref = 1 + w_cat + w_fab + w_sea

        # Sensibilidad al precio
        sensitivity = self.PROFILES[perfil]["price_sensitivity"]
        price_penalty = np.exp(-sensitivity * df_ropa["current_price"] / df_ropa["base_price"].max())

        # Factor de descuento
        discount_factor = 1 + df_ropa.get("discount", 0.0) * self.PROFILES[perfil]["promo_propensity"]

        # Peso final
        weights = base_pref * price_penalty * discount_factor * df_ropa["popularity"]
        weights = weights.clip(lower=0.01)

        df_ropa["weight"] = weights
        return df_ropa[["product_id", "weight"]]

    def _weighted_random_choice(self, df, weight_column, k):
        w = df[weight_column].values
        w = w / w.sum()
        indices = np.random.choice(len(df), size=k, replace=False, p=w)
        return df.iloc[indices]

    # -------------------------------
    # SIMULACIÓN DE COMPRA
    # -------------------------------

    def simulate_customer_purchase(self, cliente, dia, fecha_actual):
        perfil = cliente["perfil_asignado"]
        p_buy = self.PROFILES[perfil]["p_buy"]
        lambda_items = self.PROFILES[perfil]["lambda_items"]

        if random.random() > p_buy:
            return

        num_items = max(1, np.random.poisson(lambda_items))

        if dia % 6 == 0 or dia % 7 == 0:
            num_items = int(num_items * self.PROFILES[perfil]["weekend_multiplier"])

        df_stock = self._merge_items_with_ropa()
        df_stock = df_stock[df_stock["size"] == cliente["size"]]
        if df_stock.empty:
            return

        # Calcular pesos
        df_weights = self._compute_product_weights(cliente)
        df_stock = df_stock.merge(df_weights, on="product_id", how="left")

        # Simulación de items probados
        num_items = min(num_items, len(df_stock))
        pruebas = self._weighted_random_choice(df_stock, "weight", min(random.randint(2, 8), len(df_stock)))
        self.df_items.loc[self.df_items["item_id"].isin(pruebas["item_id"]), "times_tried"] += 1

        # Simulación de items comprados
        compras = self._weighted_random_choice(df_stock, "weight", num_items)

        # Cambiar status a sold
        self.df_items.loc[self.df_items["item_id"].isin(compras["item_id"]), "status"] = "Sold"

        # -------------------------
        # Generar df_purchase y df_purchase_item
        # -------------------------
        session_id = str(uuid.uuid4())
        total_amount = compras["current_price"].sum()

        # Purchase
        purchase_record = {
            "purchase_id": str(uuid.uuid4()),
            "customer_id": cliente["customer_id"],
            "session_id": session_id,
            "total_amount": total_amount,
            "timestamp": fecha_actual
        }
        self.df_purchase = pd.concat([self.df_purchase, pd.DataFrame([purchase_record])], ignore_index=True)

        # PurchaseItem
        purchase_items = []
        for _, item in compras.iterrows():
            purchase_items.append({
                "purchase_id": purchase_record["purchase_id"],
                "customer_id": cliente["customer_id"],
                "item_id": item["item_id"],
                "recommendation_id": None,
                "promotion_id": None,
                "unit_price": item["current_price"],
                "discount": item["discount"]
            })
        self.df_purchase_item = pd.concat([self.df_purchase_item, pd.DataFrame(purchase_items)], ignore_index=True)


    # -------------------------------
    # SIMULACIÓN DIARIA
    # -------------------------------

    def simular_dia(self, dia, prob_descuento=0.20):
        fecha_actual = self.start_date + timedelta(days=dia - 1)

        # Aplicar descuentos con probabilidad configurable
        self.df_ropa["discount"] = 0.0

        mask_descuento = np.random.rand(len(self.df_ropa)) < prob_descuento
        self.df_ropa.loc[mask_descuento, "discount"] = np.random.uniform(0.2, 0.8, mask_descuento.sum())
        
        # Aplicar descuento al precio actual
        self.df_ropa["current_price"] = self.df_ropa["base_price"] * (1 - self.df_ropa["discount"])

        for _, cliente in self.df_perfiles.iterrows():
            self.simulate_customer_purchase(cliente, dia, fecha_actual)


    # -------------------------------
    # SIMULACIÓN COMPLETA
    # -------------------------------

    def run_simulation(self, n_dias=30, items_per_size_category=5, prob_descuento=0.20):
        self.restock_inicial(items_per_size_category)

        for dia in range(1, n_dias + 1):
            self.restock_semanal(dia, items_per_size_category)
            self.simular_dia(dia, prob_descuento=prob_descuento)

        return self.df_purchase, self.df_purchase_item, self.df_items