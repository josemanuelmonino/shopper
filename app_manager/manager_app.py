import streamlit as st
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from db_manager import DataManagerSimple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# -------- CONFIG --------
st.set_page_config(page_title="Shopper - Manager", layout="wide")

# -------- DB ----------
dm = DataManagerSimple("database/shopper.db")

# -------- CSS ----------
with open("app_manager\style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------- SESSION STATE ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "df_clientes" not in st.session_state:
    st.session_state.df_clientes = dm.read_df("CustomerInfo")
if "df_perfiles" not in st.session_state:
    st.session_state.df_perfiles = dm.read_df("CustomerProfile")
if "df_ropa" not in st.session_state:
    st.session_state.df_ropa = dm.read_df("Product")
if "df_items" not in st.session_state:
    st.session_state.df_items = dm.read_df("Item")
if "df_recomms" not in st.session_state:
    st.session_state.df_recomms = dm.read_df("Recommendation")
if "df_proms" not in st.session_state:
    st.session_state.df_proms = dm.read_df("Promotion")
if "df_purchase" not in st.session_state:
    st.session_state.df_purchase = dm.read_df("Purchase")
if "df_purchase_items" not in st.session_state:
    st.session_state.df_purchase_items = dm.read_df("PurchaseItem")
if "df_heatmaps" not in st.session_state:
    st.session_state.df_heatmaps = dm.read_df("CustomerHeatmap")

# -------- LOGIN SIMPLE --------
def login():
    st.title("Manager Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin":
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

# -------- HEADER ----------
logo_path = r"data/raw/shopper-high-resolution-logo-transparent.png"
hcol1, hcol2 = st.columns([1, 5])

with hcol1:
    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)  # espacio arriba
    st.image(logo_path, width=300)
    st.subheader("Manager Platform")


with hcol2:
    st.write("")

st.markdown("---")

left_col, _ = st.columns([1, 3])

with left_col:
    if not st.session_state.logged_in:
        login()

if st.session_state.logged_in:
    ...