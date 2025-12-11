import pandas as pd
import random

# Crear 20 productos directamente
productos = []

for i in range(16):
    # Determinar categoría según el índice
    if i < 4:
        categoria = 'T-Shirt'
    elif i < 8:
        categoria = 'Sweater'
    elif i < 12:
        categoria = 'Pants'
    else:
        categoria = 'Coat'
    
    # Género alternado: F, F, M, M para cada grupo de 4
    if i % 4 < 2:
        genero = 'F'
    else:
        genero = 'M'
    
    # Materiales alternados
    materiales_opciones = ['Cotton', 'Wool', 'Polyester']
    material = materiales_opciones[i % 3]
    
    # Temporadas alternadas
    temporadas_opciones = ['Winter', 'Summer', 'All seasons']
    temporada = temporadas_opciones[i % 3]
    
    genero_texto = 'Women' if genero == 'F' else 'Men'
    nombre = f"{genero_texto}'s {categoria} {material}"
    
    # Precio aleatorio 20-120€
    precio_base = random.randint(20, 120) + random.choice([0.99, 0.00])
    
    # Coste (40-60% del precio)
    costo = round(precio_base * random.uniform(0.4, 0.6), 2)
    
    # Popularidad 0.2-0.7
    popularidad = round(random.uniform(0.2, 0.7), 2)
    
    # Descripción en inglés
    desc = f"{categoria} made of {material.lower()}. Perfect for {temporada.lower()}."
    
    # URL imagen
    urls = [
        "https://static.zara.net/assets/public/2730/6800/b4c746a587ee/851855d12eb4/04174660250-000-e1/04174660250-000-e1.jpg?ts=1761670352394&w=453",
        "https://static.zara.net/assets/public/2b41/5fda/0a9a4066a00f/ea9d98a2bc2f/03039453712-e1/03039453712-e1.jpg?ts=1758271030901&w=750",
        "https://static.zara.net/assets/public/c8ab/6fd4/612c439cb1e1/0da2e07f82d4/05372366250-e1/05372366250-e1.jpg?ts=1764933682273&w=358",
        "https://static.zara.net/assets/public/9b49/97fa/b4b248009c46/152b6385ccfc/05584390250-e1/05584390250-e1.jpg?ts=1756720751402&w=358",
        "https://static.zara.net/assets/public/60a7/f246/4c9c4c3086d4/0cb9a0ed1804/02756157800-e1/02756157800-e1.jpg?ts=1764666932034&w=750",
        "https://static.zara.net/assets/public/1878/8adc/46484af986e6/31a0f7452c4a/02142161644-060-e1/02142161644-060-e1.jpg?ts=1761561416820&w=750",
        "https://static.zara.net/assets/public/f20e/2f79/65874ecaad47/74b5592294c1/03284331722-e1/03284331722-e1.jpg?ts=1759746894678&w=358",
        "https://static.zara.net/assets/public/1d2d/10d7/ce984593ac35/28b0e7216cfd/02893354737-e1/02893354737-e1.jpg?ts=1764844041972&w=358",
        "https://static.zara.net/assets/public/72fa/4a41/abf94d84b300/8e60761a5ddc/09790869700-e1/09790869700-e1.jpg?ts=1763486973958&w=750",
        "https://static.zara.net/assets/public/94f3/1430/97c34d66bc23/5f16291d7b64/09649926803-e1/09649926803-e1.jpg?ts=1762866685714&w=750",
        "https://static.zara.net/assets/public/9668/2c14/5ed341bdbb7b/1d62525303e1/05854180800-e1/05854180800-e1.jpg?ts=1760364389760&w=358",
        "https://static.zara.net/assets/public/c6e6/f52a/721c40ed8688/1149e55cdd91/00761440803-000-e1/00761440803-000-e1.jpg?ts=1754985841318&w=358",
        "https://static.zara.net/assets/public/8e1a/e1b6/007746f6a3ad/9a7b78bdf456/03811244751-e1/03811244751-e1.jpg?ts=1754996157708&w=750",
        "https://static.zara.net/assets/public/1c66/1086/fd4a46b8a659/355d6d147bb0/08471107800-e1/08471107800-e1.jpg?ts=1759475177884&w=750",
        "https://static.zara.net/assets/public/1a9d/28ad/bf2c47b2a05a/96894f5fe1e9/08281104800-e1/08281104800-e1.jpg?ts=1764775833277&w=358",
        "https://static.zara.net/assets/public/e705/5ca9/783544b0a8bf/942e5c1fac7e/04192031706-e1/04192031706-e1.jpg?ts=1765287405431&w=358"
    ]
    
    # Location en inglés
    ubicaciones = ['Front display', 'Main aisle', 'Back wall']
    location = ubicaciones[i % 3]
    
    producto = {
        'product_id': f"PRD-{i+1:05d}",
        'name': nombre,
        'category': categoria,
        'gender': genero,
        'fabric': material,
        'season': temporada,
        'base_price': round(precio_base, 2),
        'current_price': round(precio_base, 2),
        'cost': costo,
        'popularity': popularidad,
        'description': desc,
        'location': location,
        'url_image': urls[i]
    }
    
    productos.append(producto)

# DataFrame y guardar
df = pd.DataFrame(productos)
df.to_csv('productos_final.csv', index=False, sep=';', encoding='utf-8')