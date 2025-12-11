import streamlit as st
import json
from collections import defaultdict

# -----------------------
# Comprobar login
# -----------------------
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("⚠️ Debes iniciar sesión para ver esta página.")
    st.stop()

# -----------------------
# Inicializar carrito si no existe
# -----------------------
if "cart" not in st.session_state:
    st.session_state.cart = []

# -----------------------
# Cargar productos
# -----------------------
with open("data/productos.json", encoding="utf-8") as f:
    productos = json.load(f)

# -----------------------
# Agrupar productos por tipo
# -----------------------
productos_agrupados = defaultdict(list)
for p in productos:
    productos_agrupados[p.get("Id_Producto_Tipo","desconocido")].append(p)

st.title("🛍️ Productos disponibles")

for tipo, items in productos_agrupados.items():
    ejemplo = items[0]  # Tomamos info de un ejemplo

    # Mostrar nombre e imagen
    st.subheader(ejemplo.get("nombre","Prenda"))
    imagen_ruta = ejemplo.get("imagen")
    if isinstance(imagen_ruta, str) and imagen_ruta.strip() != "":
        try:
            st.image(imagen_ruta, width=200)
        except Exception as e:
            st.warning(f"No se pudo cargar la imagen: {imagen_ruta}\nError: {e}")

    st.write(f"- Tipo de producto: {tipo}")
    st.write(f"- Precio: {ejemplo.get('precio',0)} €")
    st.write(f"- {ejemplo.get('descripcion','')}")

    # Selector de tallas sin duplicados
    tallas = list({i.get("Talla","N/A") for i in items})
    tallas.sort()
    seleccion = st.selectbox("Selecciona talla", tallas, key=f"{tipo}_tallas")

    # Primer item disponible de la talla seleccionada
    item_seleccionado = next(i for i in items if i.get("Talla") == seleccion)

    # Botón añadir al carrito con key único
    if st.button(f"Añadir al carrito ({seleccion})", key=item_seleccionado.get("Id_Item_Individual","producto")):
        st.session_state.cart.append(item_seleccionado)
        st.success(f"{ejemplo.get('nombre')} talla {seleccion} añadido al carrito")