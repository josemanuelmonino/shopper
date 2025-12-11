import pandas as pd
import random

class ItemGenerator:
    def __init__(self, df_ropa):
        """
        df_ropa: DataFrame con los tipos de producto (debe contener 'product_id')
        """
        self.df_ropa = df_ropa
        self.tallas = ['S', 'M', 'L', 'XL']
        self.global_counter = None

    def _init_counter(self, df_items_existente):
        """
        Inicializa el contador basado en los IDs existentes.
        Extrae el número de ITEM-XXXX.
        """
        if df_items_existente.empty:
            self.global_counter = 1
        else:
            last_num = (
                df_items_existente['item_id']
                .str.extract(r'ITEM-(\d+)')
                .astype(int)
                .max()[0]
            )
            self.global_counter = last_num + 1

    def restock(self, df_items_existente, items_per_size_category=5):
        """
        Genera ítems nuevos respetando el formato SQL:
        item_id (TEXT), product_id (TEXT), rfid_tag NULL, current_location NULL, times_tried 0
        """

        self._init_counter(df_items_existente)

        nuevos_items = []

        for _, row in self.df_ropa.iterrows():
            product_id = row['product_id']

            sizes = self.tallas * items_per_size_category
            random.shuffle(sizes)

            for talla in sizes:
                item_id = f"ITEM-{self.global_counter:04d}"

                nuevos_items.append({
                    'item_id': item_id,
                    'product_id': product_id,
                    'rfid_tag': item_id+"-TAG",
                    'size': talla,
                    'current_location': None,
                    'times_tried': 0,
                    'status': "Available"
                })

                self.global_counter += 1

        df_new = pd.DataFrame(nuevos_items)
        return pd.concat([df_items_existente, df_new], ignore_index=True)