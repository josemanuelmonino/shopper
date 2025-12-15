import streamlit as st
from streamlit_webrtc import webrtc_streamer
from streamlit_autorefresh import st_autorefresh
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from db_manager import DataManagerSimple
import time
import pandas as pd
import uuid
import queue
from src.ml_analytics.facial_emotion_engine import FaceEmotionProcessor, EmotionLiveProcessor

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

if "cand_id" not in st.session_state:
    st.session_state.cand_id = None
if "cand_hits" not in st.session_state:
    st.session_state.cand_hits = 0

if not st.session_state.df_clientes.empty:
    # Tomamos el último número, ignorando el prefijo
    last_id_num = st.session_state.df_clientes["customer_id"].str.replace("CUST-", "").astype(int).max()
else:
    last_id_num = 0

# -------- CSS ----------
with open("app_customer\style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# -------- FUNCTIONS ----------
def mostrar_productos(df, gender=None, tallas=["S","M","L","XL"], is_prom=False, is_recom=False):
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
        if is_prom:
            st.markdown(
                f"""
                <h3 style="margin-bottom: 0px;">
                    {row['name']} —
                    <span style="color: green;">${row['discount_price']:.2f}</span>
                    <span style="text-decoration: line-through; color: grey; margin-left: 6px;">
                        ${row['current_price']:.2f}
                    </span>
                </h3>
                """,
                unsafe_allow_html=True
            )
        else:
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

            if disponible.empty:
                st.warning(f"No available items for {row['name']} size {selected_size}.")
            else:
                # Tomamos el primer item disponible
                item_seleccionado = disponible.iloc[0]
                st.session_state.cart.append([item_seleccionado, row, is_recom, is_prom])
                st.success(f"Added {row['name']} size {selected_size} to cart!")

def mostrar_carrito():
    """
    Muestra los productos en el carrito con imagen, nombre, precio (tachado si hay descuento)
    y un botón para quitar cada producto del carrito.
    """
    if "cart" not in st.session_state or len(st.session_state.cart) == 0:
        st.info("Your cart is empty.")
        return

    for idx, item in enumerate(st.session_state.cart):
        # Precio
        if item[3]:  # is_prom
            st.markdown(
                f"""
                <h3 style="margin-bottom: 0px;">
                    {item[1]['name']} — Size {item[0]['size']} —
                    <span style="color: green;">${item[1]['discount_price']:.2f}</span>
                    <span style="text-decoration: line-through; color: grey; margin-left: 6px;">
                        ${item[1]['current_price']:.2f}
                    </span>
                </h3>
                """,
                unsafe_allow_html=True
            )
        else:
            st.subheader(f"{item[1]['name']} — Size {item[0]['size']} — ${item[1]['current_price']:.2f}")

        # Imagen centrada
        if 'url_image' in item[1] and pd.notna(item[1]['url_image']) and item[1]['url_image']:
            st.markdown(
                f'''
                <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                    <img src="{item[1]["url_image"]}" width="200" style="border-radius: 15px;">
                </div>
                ''',
                unsafe_allow_html=True
            )

        # Botón quitar del carrito
        remove_key = f"remove_{idx}"
        if st.button("Remove from Cart", key=remove_key):
            del st.session_state.cart[idx]
            st.rerun()

def mostrar_resumen_carrito():
    if not st.session_state.cart:
        st.info("Cart is empty.")
        return
    
    total = 0.0
    for idx, item in enumerate(st.session_state.cart):
        # Precio según si hay promoción
        precio = item[1]['discount_price'] if item[3] else item[1]['current_price']
        total += precio
        
        st.markdown(f"- {item[1]['name']} | Size: {item[0]['size']} | Price: ${precio:.2f}")

    st.subheader(f"**Total: ${total:.2f}**")
    
    if st.button("BUY"):
        # Generar purchase_id único
        purchase_id = str(uuid.uuid4())
        customer_id = st.session_state.user["customer_id"]
        session_id = st.session_state.session_id

        # Calcular total_amount
        total_amount = 0
        for _, row, is_recom, is_prom in st.session_state.cart:
            price = row["discount_price"] if is_prom else row["current_price"]
            total_amount += price

        # --- Agregar a df_purchase ---
        new_purchase = pd.DataFrame([{
            "purchase_id": purchase_id,
            "customer_id": customer_id,
            "session_id": session_id,
            "total_amount": total_amount,
            "timestamp": time.strftime("%Y-%m-%d")
        }])

        # --- Agregar a df_purchase_item ---
        items_list = []
        for item, row, is_recom, is_prom in st.session_state.cart:
            unit_price = row["discount_price"] if is_prom else row["current_price"]
            discount = row["current_price"] - unit_price if is_prom else 0.0

            # Buscar recommendation_id si aplica
            recom_id = 0
            if is_recom:
                recom_row = st.session_state.df_user_recs[
                    st.session_state.df_user_recs["product_id"] == row["product_id"]
                ]
                if not recom_row.empty:
                    recom_id = recom_row.iloc[0]["recommendation_id"]

            # Buscar promotion_id si aplica
            prom_id = 0
            if is_prom:
                prom_row = st.session_state.df_user_proms[
                    st.session_state.df_user_proms["product_id"] == row["product_id"]
                ]
                if not prom_row.empty:
                    prom_id = prom_row.iloc[0]["promotion_id"]

            items_list.append({
                "purchase_id": purchase_id,
                "item_id": item["item_id"],
                "customer_id": customer_id,
                "recommendation_id": recom_id,
                "promotion_id": prom_id,
                "unit_price": unit_price,
                "discount": discount
            })

        new_purchase_items = pd.DataFrame(items_list)

        # Actualizar df_items: cambiar status de los items comprados
        df_items = st.session_state.df_items.copy()
        df_items.loc[df_items["item_id"].isin([item["item_id"] for item, _, _, _ in st.session_state.cart]), "status"] = "Sold"

        # Guardar en la DB
        dm.save_df(new_purchase, "Purchase", if_exists="append")
        dm.save_df(new_purchase_items, "PurchaseItem", if_exists="append")
        dm.save_df(df_items, "Item")

        st.session_state.df_items = df_items
        st.session_state.cart.clear()
        st.success("Purchase completed! Cart is now empty.")

        st.rerun()

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
# -------- MAIN LAYOUT: 3 columnas ----------
left_col, center_col, right_col = st.columns([2, 3, 2])

# -------- CENTER: Cámara ----------
with center_col:
    
    # CASO 1: NO LOGUEADO -> MOSTRAR CÁMARA Y BUSCAR
    if not st.session_state.logged_in:
        st.write("### Face Login System")
        
        # 1. INICIAR EL STREAMER (Solo si no estamos logueados)
        ctx = webrtc_streamer(
            key="live_cam",
            video_processor_factory=FaceEmotionProcessor,
            media_stream_constraints={
                "video": {"width": {"min": 1280, "ideal": 1920}, "height": {"min": 720, "ideal": 1080}},
                "audio": False
            },
            async_processing=True,
            desired_playing_state=True,
        )

        # Auto-refresh para mantener el bucle vivo mientras busca
        if getattr(ctx.state, "playing", False):
            st_autorefresh(interval=500, key="cam_poll")

        # 2. LÓGICA DE PROCESAMIENTO DE COLA (Tu código corregido)
        if ctx.video_processor:
            try:
                while True:
                    detected_name = ctx.video_processor.result_queue.get_nowait()
                    detected_clean = str(detected_name).strip()
                    
                    # Feedback visual
                    st.toast(f"Detected: {detected_clean} (Hits: {st.session_state.cand_hits})")

                    if st.session_state.cand_id == detected_clean:
                        st.session_state.cand_hits += 1
                    else:
                        st.session_state.cand_id = detected_clean
                        st.session_state.cand_hits = 0

                    # --- LOGIN ÉXITO ---
                    if st.session_state.cand_hits >= 1:
                        st.session_state.df_clientes = dm.read_df("CustomerInfo")
                        dfc = st.session_state.df_clientes.copy()

                        # Búsqueda por ID o por nombre
                        user_match = dfc[dfc["customer_id"].astype(str) == detected_clean]
                        if user_match.empty:
                            name_search = detected_clean.replace("_", " ").lower()
                            user_match = dfc[dfc["name"].astype(str).str.lower().str.contains(name_search)]

                        if not user_match.empty:
                            st.session_state.user = user_match.iloc[0]
                            st.session_state.logged_in = True
                            st.session_state.session_id = str(uuid.uuid4())

                            # IMPORTANTE: cargar recs/promos al loguear (esto viene de tu versión)
                            customer_id = st.session_state.user["customer_id"]
                            st.session_state.df_user_recs = st.session_state.df_recomms[
                                st.session_state.df_recomms["customer_id"] == customer_id
                            ]
                            st.session_state.df_user_proms = st.session_state.df_proms[
                                st.session_state.df_proms["customer_id"] == customer_id
                            ]

                            st.success(f"Login successful: {st.session_state.user['name']}")
                            st.rerun()
                        else:
                            st.error(f"User '{detected_clean}' recognized but not in DB.")
                            st.session_state.cand_hits = 0
                            
            except queue.Empty:
                pass
                
        # 3. MENÚ DE REGISTRO FACIAL 
        st.markdown("---")
        with st.expander("📸 Create Facial Profile (Register)", expanded=False):
            st.write("To register, enter your details first.")
            
            # Formulario rápido para generar el ID antes de la foto
            reg_name = st.text_input("Full Name:", key="reg_face_name")
            reg_email = st.text_input("Email:", key="reg_face_email")
            reg_gender = st.radio("Gender:", ["F", "M"], horizontal=True, key="reg_face_gender")
            
            col_btn1, col_btn2 = st.columns(2)
            
            # BOTÓN 1: Crear Usuario + Iniciar Captura
            if col_btn1.button("▶ Create User and Start Capture"):
                if reg_name and reg_email and ctx.video_processor:
                    try:
                        # 1. Calcular nuevo ID (CUST-XXXX)
                        if not st.session_state.df_clientes.empty:
                            last_id = st.session_state.df_clientes["customer_id"].str.replace("CUST-", "").astype(int).max()
                            new_id_num = last_id + 1
                        else:
                            new_id_num = 1
                        
                        new_cust_id = f"CUST-{new_id_num:05d}"
                        
                        # 2. Guardar en DB (Provisionalmente sin contraseña o con una default)
                        new_customer = pd.DataFrame([{
                            "customer_id": new_cust_id, 
                            "name": reg_name, 
                            "email": reg_email, 
                            "pass": "1234", # Contraseña por defecto o pide input
                            "gender": reg_gender
                        }])
                        
                        dm.save_df(new_customer, "CustomerInfo", if_exists="append")
                        # Recargar df local
                        st.session_state.df_clientes = dm.read_df("CustomerInfo")
                        
                        st.success(f"User created: {new_cust_id}. Look at the camera.")
                        
                        # 3. Iniciar Captura usando el ID como nombre de carpeta
                        ctx.video_processor.start_capture_sequence(
                            name=reg_name, 
                            customer_id_folder=new_cust_id  # USA EL ID COMO NOMBRE DE CARPETA
                        )
                        
                    except Exception as e:
                        st.error(f"Error creating user: {e}")
                        
                elif not ctx.video_processor:
                    st.warning("Please start the camera first (above).")
                else:
                    st.warning("Fill in name and email.")

            # BOTÓN 2: Siguiente Fase 
            if col_btn2.button("⏭ Next Phase / Finish"):
                if ctx.video_processor:
                    ctx.video_processor.continue_next_phase()

    # ==============================================================================
    # CASO 2: LOGUEADO -> PERFIL + (CÁMARA DE EMOCIONES O CÁMARA DE REGISTRO)
    # ==============================================================================
    else:
        st.success("Identity Verified")
        
        col_info, col_cam = st.columns([1, 2])
        
        with col_info:
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h3>Hello, {st.session_state.user['name']}</h3>
                <p><strong>ID:</strong> {st.session_state.user['customer_id']}</p>
                <p>Welcome to the customer experience analysis system.</p>
                <div style="font-size: 40px; margin-top:10px;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- INTERRUPTOR DE MODO ---
            st.write("**Camera Settings:**")
            # Si se activa, update_mode será True
            update_mode = st.toggle("📸 Update/Register Facial Photos")
            
            st.markdown("---")
            if st.button("Log Out"):
                st.session_state.logged_in = False
                st.session_state.user = None
                st.session_state.cand_hits = 0
                st.rerun()

        with col_cam:
            
            # ------------------------------------------------------------------
            # MODO A: REGISTRO/ACTUALIZACIÓN DE FOTOS (FaceEmotionProcessor)
            # ------------------------------------------------------------------
            if update_mode:
                st.write("### 📸 Update Biometric Data")
                st.warning("New photos will be added to your ID folder.")
                
                # Usamos el procesador de captura
                ctx_update = webrtc_streamer(
                    key="update_face_cam", # Key diferente para forzar recarga
                    video_processor_factory=FaceEmotionProcessor,
                    media_stream_constraints={
                        "video": {"width": 1280, "height": 720}, 
                        "audio": False
                    },
                    async_processing=True,
                    desired_playing_state=True,
                )

                # Controles de captura (Solo si la cámara está activa)
                if ctx_update.video_processor:
                    col_b1, col_b2 = st.columns(2)
                    
                    if col_b1.button("▶ Start Capture (Add Photos)"):
                        # AQUÍ ESTÁ LA CLAVE: Usamos el ID del usuario logueado
                        current_id = st.session_state.user['customer_id']
                        current_name = st.session_state.user['name']
                        
                        ctx_update.video_processor.start_capture_sequence(
                            name=current_name,
                            customer_id_folder=current_id # <--- Guarda en SU carpeta existente
                        )
                        st.toast(f"Capturing for: {current_id}")

                    if col_b2.button("⏭ Next Phase"):
                        ctx_update.video_processor.continue_next_phase()
            
            # ------------------------------------------------------------------
            # MODO B: ANÁLISIS EMOCIONAL (EmotionLiveProcessor) - DEFAULT
            # ------------------------------------------------------------------
            else:
                st.write("### Real-Time Emotional Analysis")
                st.info("Monitoring Satisfaction and Frustration levels...")
                
                # Usamos el procesador de emociones
                ctx_emotion = webrtc_streamer(
                    key="emotion_cam",
                    video_processor_factory=EmotionLiveProcessor,
                    media_stream_constraints={
                        "video": {"width": 1024, "height": 786}, 
                        "audio": False
                    },
                    async_processing=True,
                    desired_playing_state=True
                )

                # Pasamos datos al procesador de emociones
                if ctx_emotion.video_processor:
                    ctx_emotion.video_processor.update_user_info(
                        customer_id=st.session_state.user['customer_id'],
                        session_id=st.session_state.get("session_id", "Unknown")
                    )
                    
# -------- LEFT PANEL ----------
with left_col:
    if st.session_state.active_tab == "Home":
        st.header(f"{st.session_state.active_tab}")
        st.markdown("""
            **Welcome to the SHOPPER project!** The SHOPPER project proposes an innovative system that transforms the in-store shopping experience through the integration of advanced technologies such as RFID, BLE tracking, smart mirrors, and Emotion AI. By combining real-time data analysis with personalized recommendations and dynamic pricing, SHOPPER enables fashion retailers to optimize store management, improve customer satisfaction, and bridge the gap between physical and digital retail environments.

            **Created at the UPM** by Lucia Pintos, José Moñino, Cristina García-Yañez, and Miguel Ángel Conde.
            
            **GitHub**: https://github.com/josemanuelmonino/shopper
                """)
        
    elif st.session_state.get("logged_in"):
        if st.session_state.active_tab == "Shop":
            st.subheader("**WOMEN CLOTHING**")
            mostrar_productos(st.session_state.df_ropa, gender="F")

        elif st.session_state.active_tab == "Recommendations & Promotions":
            st.subheader("**RECOMMENDATIONS**")
            if st.session_state.df_user_recs.empty:
                st.info("You have no recommendations yet.")
            else:
                recommended_products = st.session_state.df_user_recs["product_id"]

                df_recommended_products = st.session_state.df_ropa[
                    st.session_state.df_ropa["product_id"].isin(recommended_products)
                ]
                mostrar_productos(df_recommended_products, is_recom=True)

        else: #Cart
            st.subheader("**BUY**")
            st.markdown("---")
            mostrar_resumen_carrito()


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
        if st.session_state.active_tab == "Shop":
            st.subheader("**MEN CLOTHING**")
            mostrar_productos(st.session_state.df_ropa, gender="M")

        elif st.session_state.active_tab == "Recommendations & Promotions":
            st.subheader("**PROMOTIONS**")
            if st.session_state.df_user_proms.empty:
                st.info("You have no promotions yet.")
            else:
                proms = st.session_state.df_user_proms[["product_id", "discount_percentage"]]

                df_promoted_products = (
                    st.session_state.df_ropa
                        .merge(proms, on="product_id", how="inner")
                )

                df_promoted_products["discount_price"] = (
                    df_promoted_products["current_price"] * df_promoted_products["discount_percentage"]
                )

                mostrar_productos(df_promoted_products, is_prom=True)

        else: #Cart
            st.subheader("**CART**")
            st.markdown("---")
            mostrar_carrito()
            
