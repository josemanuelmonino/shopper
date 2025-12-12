import streamlit as st
from streamlit_webrtc import webrtc_streamer
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from db_manager import DataManagerSimple
import pandas as pd
import uuid

# -------- CONFIG ----------
st.set_page_config(page_title="Shopper", layout="wide")

# -------- DB ----------
dm = DataManagerSimple("database/shopper.db")

# -------- SESSION STATE ----------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"
if "show_login_form" not in st.session_state:
    st.session_state.show_login_form = False
if "show_sign_form" not in st.session_state:
    st.session_state.show_sign_form = False
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "df_clientes" not in st.session_state:
    st.session_state.df_clientes = dm.read_df("CustomerInfo")
if "df_ropa" not in st.session_state:
    st.session_state.df_ropa = dm.read_df("Product")
if "df_items" not in st.session_state:
    st.session_state.df_items = dm.read_df("Item")
if "df_recomms" not in st.session_state:
    st.session_state.df_recomms = dm.read_df("Recommendation")
if "df_proms" not in st.session_state:
    st.session_state.df_proms = dm.read_df("Promotion")
if "cart" not in st.session_state:
    st.session_state.cart = []

if not st.session_state.df_clientes.empty:
    # Tomamos el último número, ignorando el prefijo
    last_id_num = st.session_state.df_clientes["customer_id"].str.replace("CUST-", "").astype(int).max()
else:
    last_id_num = 0

# -------- CSS ----------
with open("app\style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# -------- FUNCTIONS ----------
def mostrar_productos(df, gender=None, tallas=["S","M","L","XL"]):
    """
    Muestra todos los productos filtrados por género con nombre, current_price, descripción, imagen,
    selector de talla y botón para añadir al carrito. Selecciona un item disponible de df_items.
    
    df: DataFrame con columnas name, current_price, description, url_image, gender, product_id
    gender: 'F', 'M' o None (None muestra todos)
    tallas: lista de tallas disponibles
    """
    productos_filtrados = df.copy()
    
    if gender is not None:
        productos_filtrados = productos_filtrados[productos_filtrados['gender'] == gender]

    for idx, row in productos_filtrados.iterrows():
        st.markdown("---")  # Separador entre productos
        st.subheader(f"{row['name']} — ${row['current_price']:.2f}")
        
        # Imagen centrada y con bordes redondeados
        if 'url_image' in row and pd.notna(row['url_image']) and row['url_image']:
            st.markdown(
                f'''
                <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                    <img src="{row["url_image"]}" width="200" style="border-radius: 15px;">
                </div>
                ''',
                unsafe_allow_html=True
            )
        # Descripción
        if 'description' in row and pd.notna(row['description']):
            st.write(row['description'])

        # Selector de talla
        talla_key = f"size_{row['product_id']}"
        selected_size = st.selectbox("Select size", options=tallas, key=talla_key)

        # Botón añadir al carrito
        add_key = f"add_{row['product_id']}_{selected_size}"
        if st.button("Add to Cart", key=add_key):
            # Buscar un item disponible en st.session_state.df_items
            df_items = st.session_state.df_items
            disponible = df_items[
                (df_items['product_id'] == row['product_id']) &
                (df_items['size'] == selected_size) &
                (df_items['status'] == "Available")
            ]

            if df_items.empty:
                st.warning(f"No available items for {row['name']} size {selected_size}.")
            else:
                # Tomamos el primer item disponible
                item_seleccionado = disponible.iloc[0]
                st.session_state.cart.append(item_seleccionado["item_id"])
                st.success(f"Added {row['name']} size {selected_size} to cart!")

# -------- HEADER ----------
logo_path = r"data/raw/shopper-high-resolution-logo-transparent.png"

hcol1, hcol2 = st.columns([1, 5])

with hcol1:
    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)  # espacio arriba
    st.image(logo_path, width=400)
    if st.session_state.logged_in:
        st.markdown(f"Welcome back {st.session_state.user['name']}!")

with hcol2:
    st.write("")

    selected_tab = st.radio(
        "Select tab",
        ["Home", "Shop", "Recommendations & Promotions", "Cart"],
        horizontal=True,
        index=["Home", "Shop", "Recommendations & Promotions", "Cart"].index(st.session_state.active_tab),
        label_visibility="collapsed",
    )

    # Actualizar estado solo si cambió
    if selected_tab != st.session_state.active_tab:
        st.session_state.active_tab = selected_tab
        st.rerun()

st.markdown("---")


# -------- MAIN LAYOUT: 3 columnas ----------
left_col, center_col, right_col = st.columns([2, 3, 2])

# -------- CENTER: Cámara ----------
with center_col:
    webrtc_ctx = webrtc_streamer(
        key="live_cam",
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    if not st.session_state.logged_in:
        # --- Botón de login alternativo ---
        if st.button("Login with email"):
            st.session_state.show_login_form = not st.session_state.show_login_form

        # --- Formulario de login existente ---
        if st.session_state.show_login_form:
            st.subheader("Login with email and password")
            email = st.text_input("Email address", key="login_email", placeholder="Enter your email")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")

            if st.button("Login", key="login_btn"):
                def login_db(email, password, df_clientes):
                    user = df_clientes.loc[df_clientes["email"] == email]
                    if user.empty:
                        return None
                    if str(user["pass"].values[0]) == str(password):
                        return user.iloc[0]
                    return None

                st.session_state.user = login_db(email, password, st.session_state.df_clientes)
                st.session_state.session_id = str(uuid.uuid4())

                if st.session_state.user is not None:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Incorrect email or password")

        # --- Botón para crear cuenta / sign in ---
        if st.button("Sign in"):
            st.session_state.show_sign_form = not st.session_state.show_sign_form

        # --- Formulario de sign in / create account ---
        if st.session_state.show_sign_form:
            st.subheader("Create your own Shopper account")
            new_name = st.text_input("Full Name", key="sign_name", placeholder="Enter your full name")
            new_email = st.text_input("Email address", key="sign_email", placeholder="Enter your email")
            new_password = st.text_input("Password", type="password", key="sign_pass", placeholder="Enter a password")
            gender = st.radio(
                "Gender",
                options=["F", "M"],
                key="sign_gender",
                horizontal=True
            )
            

            if st.button("Sign in / Create", key="sign_btn"):
                # Aquí podrías añadir la lógica de creación de cuenta
                if new_email and new_password and new_name:
                    # Generar nuevo customer_id con formato CUST-00001
                    new_customer_id = f"CUST-{last_id_num + 1:05d}"

                    # Crear DataFrame del nuevo cliente
                    new_customer = pd.DataFrame([{
                        "customer_id": new_customer_id,
                        "name": new_name,
                        "email": new_email,
                        "pass": new_password,
                        "gender": gender
                    }])
                    dm.save_df(new_customer, "CustomerInfo", if_exists="append")
                    st.success(f"Account created for {new_name}! You can now log in.")
                    st.session_state.df_clientes = dm.read_df("CustomerInfo")
                else:
                    st.error("Please fill all fields")


# -------- LEFT PANEL ----------
with left_col:
    if st.session_state.active_tab == "Home":
        st.header(f"{st.session_state.active_tab}")
        st.markdown("""
            **Welcome to the SHOPPER project!**  

            The SHOPPER project proposes an innovative system that transforms the in-store shopping experience through the integration of advanced technologies such as RFID, BLE tracking, smart mirrors, and Emotion AI. By combining real-time data analysis with personalized recommendations and dynamic pricing, SHOPPER enables fashion retailers to optimize store management, improve customer satisfaction, and bridge the gap between physical and digital retail environments.

            **Created at the UPM** by Lucia Pintos, José Moñino, Cristina García-Yañez, and Miguel Ángel Conde.
                """)
        
    elif st.session_state.get("logged_in"):
        st.header(f"{st.session_state.active_tab}")

        if st.session_state.active_tab == "Shop":
            st.write("WOMEN CLOTHING")
            mostrar_productos(st.session_state.df_ropa, gender="F")

        elif st.session_state.active_tab == "Recommendations & Promotions":
            st.write("Profile left: user info.")

        else: #Cart
            st.write("Other left controls.")

# -------- RIGHT PANEL ----------
with right_col:
    if st.session_state.active_tab == "Home":
        st.header(f"{st.session_state.active_tab}")
        st.markdown("""
            **Technologies & Libraries Used:**

            - Python 3.11  
            - **DeepFace** for emotion and facial recognition  
            - **YOLOv8** for face detection 
            - **Streamlit** for the web interface
            - **MiniBatchKMeans** for clustering and analysis of customer patterns
            - **Simulated sales data** for model training and testing  
            - **LightGBM** for predictive analytics  
            - Many other libraries for data processing, visualization, and AI functionalities
        """)

    elif st.session_state.get("logged_in"):
        st.header(f"{st.session_state.active_tab}")
        if st.session_state.active_tab == "Shop":
            st.write("MEN CLOTHING")
            mostrar_productos(st.session_state.df_ropa, gender="M")

        elif st.session_state.active_tab == "Recommendations & Promotions":
            st.write("User profile info...")

        else: #Cart
            st.write("Other right info.")