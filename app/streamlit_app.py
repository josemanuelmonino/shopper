import streamlit as st
from pathlib import Path
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from db_manager import DataManagerSimple

dm = DataManagerSimple("database/shopper.db")

# -----------------------
# SESIÓN
# -----------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "cart" not in st.session_state:
    st.session_state.cart = []

# -----------------------
# Cargar usuarios reales
# -----------------------
df_clientes = dm.read_df("CustomerInfo")


# -----------------------
# Función de login DB
# -----------------------
def login_db(email, password, df_clientes):
    # ADMIN
    if email == "admin" and password == "admin":
        return "admin"

    # Buscar usuario por email
    user = df_clientes.loc[df_clientes["email"] == email]

    if user.empty:
        return None

    # Comprobar contraseña
    if str(user["pass"].values[0]) == str(password):
        return user.iloc[0]   # Devuelvo fila completa del usuario

    return None


# -----------------------
# LOGIN
# -----------------------
if not st.session_state.logged_in:

    st.title("Login - Shopper Demo")
    st.write("Introduce tus credenciales")

    email = st.text_input("Email")
    password = st.text_input("Contraseña", type="password")

    if st.button("Entrar"):

        result = login_db(email, password, df_clientes)

        if result is None:
            st.error("❌ Email o contraseña incorrectos")

        else:
            st.session_state.logged_in = True

            if result == "admin":
                st.session_state.user = {"name": "Admin", "email": "admin"}
                st.session_state.is_admin = True
                st.success("🔐 Sesión iniciada como ADMIN")
            else:
                st.session_state.user = dict(result)
                st.session_state.is_admin = False
                st.success(f"¡Bienvenido {result['name']}!")

            st.info("Ahora selecciona una página desde el menú lateral.")
else:
    st.title("Shopper Demo")
    st.write(f"Sesión iniciada como: **{st.session_state.user['name']}**")
    if st.session_state.is_admin:
        st.warning("🔐 Modo administrador activo")