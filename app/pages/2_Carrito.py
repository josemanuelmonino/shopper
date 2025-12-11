import streamlit as st

# -----------------------
# Comprobar login
# -----------------------
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("⚠️ Debes iniciar sesión para ver esta página.")
    st.stop()

st.title("🛒 Tu carrito")

# -----------------------
# Obtener carrito
# -----------------------
carrito = st.session_state.cart if "cart" in st.session_state else []

if carrito:
    total = sum(p.get("precio", 0) for p in carrito)

    for idx, p in enumerate(carrito):
        st.write(f"**{p.get('nombre','Producto')}** — {p.get('precio',0)} € (Talla {p.get('Talla','N/A')})")

        imagen_ruta = p.get("imagen")
        if isinstance(imagen_ruta, str) and imagen_ruta.strip() != "":
            try:
                st.image(imagen_ruta, width=100)
            except Exception as e:
                st.warning(f"No se pudo cargar la imagen: {imagen_ruta}\nError: {e}")
        st.markdown("---")  # Separador entre productos

    st.write(f"### Total: {total} €")

    if st.button("Vaciar carrito"):
        st.session_state.cart = []
        st.success("Carrito vaciado")
else:
    st.info("Tu carrito está vacío")