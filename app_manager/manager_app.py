import streamlit as st
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from db_manager import DataManagerSimple
import pandas as pd
import numpy as np
import json
import plotly.express as px

# -------- CONFIG --------
st.set_page_config(page_title="Shopper - Manager", layout="wide")

# -------- DB ----------
dm = DataManagerSimple("database/shopper.db")

# -------- CSS ----------
with open("app_manager\style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------- SESSION STATE ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = True
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Sales & Products"
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
if "df_heatmaps" not in st.session_state:
    st.session_state.df_heatmaps = dm.read_df("CustomerHeatmap")
if "df_emotions" not in st.session_state:
    st.session_state.df_emotions = dm.read_df("EmotionEvent")

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

# -------- FUNCTIONS --------
def prepare_sales_df():
    df = st.session_state.df_purchase_items \
        .merge(st.session_state.df_items, on="item_id", how="left") \
        .merge(st.session_state.df_ropa, on="product_id", how="left") \
        .merge(st.session_state.df_purchase, on="purchase_id", how="left")

    return df

def show_sales_products():
    st.subheader("Sales & Products")

    df = prepare_sales_df()

    # ---------- KPIs ----------
    total_sales = len(df)
    total_revenue = df["current_price"].sum()
    avg_price = df["current_price"].mean()

    k1, k2, k3 = st.columns(3)
    k1.metric("Items sold", total_sales)
    k2.metric("Revenue (€)", f"{total_revenue:.2f}")
    k3.metric("Avg item price (€)", f"{avg_price:.2f}")

    st.divider()

    # ---------- TOP PRODUCTS SOLD ----------
    st.subheader("Top Products Sold")
    top_products = (
        df.groupby(["product_id", "name"])
        .size()
        .reset_index(name="sales")
        .sort_values("sales", ascending=False)
        .head(6)
    )
    fig = px.bar(
        top_products,
        x="name",
        y="sales",
        color="sales",
        color_continuous_scale="Plasma",
        title="Top 6 Products by Units Sold"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    # ---------- WORST PRODUCTS ----------
    st.subheader("Worst Products")
    bottom_products = (
        df.groupby(["product_id", "name"])
        .size()
        .reset_index(name="sales")
        .sort_values("sales", ascending=True)
        .head(6)
    )
    fig = px.bar(
        bottom_products,
        x="name",
        y="sales",
        color="sales",
        color_continuous_scale="Plasma",
        title="Worst 6 Products by Units Sold"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    # ---------- SALES DISTRIBUTION ----------
    st.subheader("Sales Distribution")
    group_by = st.selectbox(
        "Group sales by",
        ["category", "fabric", "gender"]
    )
    grouped = (
        df.groupby(group_by)
        .size()
        .reset_index(name="sales")
        .sort_values("sales", ascending=False)
    )
    fig = px.bar(
        grouped,
        x=group_by,
        y="sales",
        color="sales",
        color_continuous_scale="Plasma",
        title=f"Sales by {group_by.capitalize()}"
    )
    st.plotly_chart(fig, use_container_width=True)

def decode_heatmap(heatmap_json):
    return np.array(json.loads(heatmap_json))

def show_heatmaps():
    st.subheader("Store Heatmaps")
    k1, k2 = st.columns(2)
    df = st.session_state.df_heatmaps

    with k1:
        customer = st.selectbox(
            "Select customer",
            df["customer_id"].unique()
        )
        row = df[df["customer_id"] == customer].iloc[-1]
        heatmap = decode_heatmap(row["heatmap"])

        fig = px.imshow(
            heatmap,
            color_continuous_scale="YlOrRd",
            origin="lower",
            title=f"Heatmap – {customer}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with k2:
        row = df[df["customer_id"] == "CUST-00000"].iloc[0]
        heatmap = decode_heatmap(row["heatmap"])

        fig = px.imshow(
            heatmap,
            color_continuous_scale="YlOrRd",
            origin="lower",
            title=f"General Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_customer_profiles():
    st.subheader("Customer Profiles")
    df = st.session_state.df_perfiles.copy()

    # ---------- KPIs ----------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total customers", len(df))
    k2.metric("Avg total spent", f"€{df['total_spent'].mean():.2f}")
    k3.metric("Avg total purchases", f"{df['total_purchases'].mean():.2f}")
    k4.metric("Avg discount", f"{df['average_discount'].mean()*100:.2f}%")
    
    st.divider()

    # ---------- TOP SPENDERS ----------
    st.subheader("Top 10 Customers by Total Spent")
    top_spenders = df.sort_values("total_spent", ascending=False).head(10)
    fig = px.bar(
        top_spenders,
        x="customer_id",
        y="total_spent",
        color="total_spent",
        color_continuous_scale="Viridis",
        title="Top 10 Customers by Total Spent"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    # ---------- CATEGORY & FABRIC PREFERENCE ----------
    k1, k2 = st.columns(2)
    with k1:
        st.subheader("Category Preference")
        category_counts = df["category_preference"].value_counts().reset_index()
        category_counts.columns = ["category", "count"]
        fig = px.pie(
            category_counts,
            names="category",
            values="count",
            title="Customers by Category Preference"
        )
        st.plotly_chart(fig, use_container_width=True)

    with k2:
        st.subheader("Fabric Preference")
        fabric_counts = df["fabric_preference"].value_counts().reset_index()
        fabric_counts.columns = ["fabric", "count"]
        fig = px.pie(
            fabric_counts,
            names="fabric",
            values="count",
            title="Customers by Fabric Preference"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()

    # ---------- CLUSTER OVERVIEW ----------
    st.subheader("Customer Cluster Overview")
    fig = px.scatter(
        df,
        x="total_purchases",
        y="total_spent",
        color=df["cluster"].astype(str),
        hover_data=["customer_id"],
        title="Customer Cluster: Purchases vs Spending",
        size=[10]*len(df),  # tamaño uniforme más grande
        size_max=15
    )
    st.plotly_chart(fig, use_container_width=True)

def parse_emotion_json(json_str, target_emotion):
    """Extrae el valor de una emoción específica del string JSON almacenado en la BD."""
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            return data.get(target_emotion, 0.0)
        return 0.0
    except:
        return 0.0

def show_emotions_alerts():
    st.subheader("Emotions & Alerts Analytics")
    
    # 1. Cargar datos
    df = st.session_state.df_emotions.copy()
    
    if df.empty:
        st.warning("No emotion events recorded in the database.")
        return

    # Asegurar formato de fecha
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 2. KPIs Globales
    # Definimos "Alerta" como Frustración > 0.5 (50%)
    frustration_events = df[df["group_label"] == "frustration"]
    high_frustration = frustration_events[frustration_events["avg_window"] > 0.5]
    
    total_alerts = len(high_frustration)
    dominant_mood = df["group_label"].mode()[0] if not df.empty else "N/A"
    avg_intensity = df["avg_window"].mean()

    k1, k2, k3 = st.columns(3)
    k1.metric("Frustration Alerts (>50%)", total_alerts, delta_color="inverse")
    k2.metric("Dominant Store Emotion", dominant_mood.capitalize())
    k3.metric("Average intensity", f"{avg_intensity:.1f}")

    st.divider()

    # 3. Gráfica de Evolución Temporal (Timeline)
    st.subheader("Customer Emotional Journey")
    
    c1, c2 = st.columns([1, 3])
    
    with c1:
        # Selectores para filtrar
        customer_list = df["customer_id"].unique()
        selected_customer = st.selectbox("Select Customer", customer_list)
        
        cust_df = df[df["customer_id"] == selected_customer].sort_values("timestamp")
        
        session_list = cust_df["session_id"].unique()
        selected_session = st.selectbox("Select Session", session_list)
        
        # Datos filtrados finales
        session_df = cust_df[cust_df["session_id"] == selected_session].copy()

    with c2:
        if not session_df.empty:
            # Transformar los datos JSON para Plotly
            plot_data = []
            target_emotions = ["frustration", "happiness", "surprise", "neutral"]
            
            for _, row in session_df.iterrows():
                t = row["timestamp"]
                json_str = row["emotions"]
                for emo in target_emotions:
                    val = parse_emotion_json(json_str, emo)
                    plot_data.append({"Time": t, "Emotion": emo.capitalize(), "Score": val})
            
            df_plot = pd.DataFrame(plot_data)

            # Colores corporativos/semánticos
            color_map = {
                "Frustration": "#FF4B4B",  # Rojo
                "Happiness": "#2ECC71",    # Verde
                "Surprise": "#3498DB",     # Azul
                "Neutral": "#95A5A6"       # Gris
            }

            fig = px.line(
                df_plot, 
                x="Time", 
                y="Score", 
                color="Emotion",
                color_discrete_map=color_map,
                title=f"Emotional Evolution - Session: {selected_session}",
                markers=True
            )
            fig.update_layout(yaxis_title="Intensity Score (0-100)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for this session.")

    st.divider()
    
    # 4. Distribución y Tabla de Alertas
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Global Emotion Distribution")
        grouped_counts = df["group_label"].value_counts().reset_index()
        grouped_counts.columns = ["Emotion", "Count"]
        
        fig_pie = px.pie(
            grouped_counts, 
            names="Emotion", 
            values="Count", 
            hole=0.4,
            color="Emotion",
            color_discrete_map={
                "frustration": "#FF4B4B", 
                "happiness": "#2ECC71", 
                "surprise": "#3498DB", 
                "neutral": "#95A5A6"
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Recent High Intensity Alerts")
        if not high_frustration.empty:
            # Preparamos tabla bonita
            alerts_display = high_frustration[["timestamp", "customer_id", "avg_window"]].copy()
            alerts_display = alerts_display.sort_values("timestamp", ascending=False).head(10)
            alerts_display.rename(columns={"avg_window": "Intensity", "customer_id": "Customer"}, inplace=True)
            
            st.dataframe(
                alerts_display, 
                hide_index=True,
                use_container_width=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Time", format="D MMM, HH:mm:ss"),
                    "Intensity": st.column_config.ProgressColumn(
                        "Frustration Level", 
                        format="%.1f", 
                        min_value=0, 
                        max_value=100  # Updated to 100 based on your data
                    )
                }
            )
        else:
            st.success("No recent frustration alerts detected.")


# -------- HEADER ----------
logo_path = r"data/raw/shopper-high-resolution-logo-transparent.png"
hcol1, hcol2 = st.columns([1, 5])

with hcol1:
    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)  # espacio arriba
    st.image(logo_path, width=300)
    st.markdown("Manager Platform")


with hcol2:
    st.write("")

st.markdown("---")

left_col, _ = st.columns([1, 3])

with left_col:
    if not st.session_state.logged_in:
        login()

if st.session_state.logged_in:
    with hcol2:
        st.write("")

        selected_tab = st.radio(
            "Select tab",
            ["Sales & Products", "Heatmaps", "Customer Profiles", "Emotions & Alerts"],
            horizontal=True,
            index=["Sales & Products", "Heatmaps", "Customer Profiles", "Emotions & Alerts"].index(st.session_state.active_tab),
            label_visibility="collapsed",
        )
        if selected_tab != st.session_state.active_tab:
            st.session_state.active_tab = selected_tab
            st.rerun()
    
    if st.session_state.active_tab == "Sales & Products":
        show_sales_products()
    elif st.session_state.active_tab == "Heatmaps":
        show_heatmaps()
    elif st.session_state.active_tab == "Customer Profiles":
        show_customer_profiles()
    else: #Emotions & Alerts
        show_emotions_alerts()