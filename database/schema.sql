-- Creación de tablas para el sistema SHOPPER

-- Tabla de información de clientes
CREATE TABLE IF NOT EXISTS CustomerInfo (
    customer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    pass TEXT DEFAULT 1234,
    gender TEXT
);

-- Tabla de perfiles de clientes
CREATE TABLE IF NOT EXISTS CustomerProfile (
    customer_id TEXT PRIMARY KEY,
    total_purchases INTEGER DEFAULT 0,
    total_spent REAL DEFAULT 0.0,
    average_discount REAL DEFAULT 0.0,
    category_preference TEXT,
    fabric_preference TEXT,
    size_preference TEXT,
    cluster INTEGER,
    FOREIGN KEY (customer_id) REFERENCES CustomerInfo(customer_id)
);

-- Tabla de productos
CREATE TABLE IF NOT EXISTS Product (
    product_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    gender TEXT,
    fabric TEXT,
    season TEXT,
    base_price REAL NOT NULL,
    current_price REAL NOT NULL,
    cost REAL NOT NULL,
    popularity REAL DEFAULT 0.0,
    description TEXT,
    location TEXT,
    url_image TEXT
);

-- Tabla de ítems físicos (instancias de productos con RFID)
CREATE TABLE IF NOT EXISTS Item (
    item_id TEXT PRIMARY KEY,
    product_id TEXT NOT NULL,
    rfid_tag TEXT UNIQUE,
    size TEXT NOT NULL,
    current_location TEXT,
    times_tried INTEGER DEFAULT 0,
    status TEXT DEFAULT "Available",
    FOREIGN KEY (product_id) REFERENCES Product(product_id)
);

-- Tabla de sesiones de compra
CREATE TABLE IF NOT EXISTS ShoppingSession (
    session_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP,
    active BOOLEAN DEFAULT 1,
    FOREIGN KEY (customer_id) REFERENCES CustomerInfo(customer_id)
);

-- Tabla de compras realizadas
CREATE TABLE IF NOT EXISTS Purchase (
    purchase_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    total_amount REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES CustomerInfo(customer_id),
    FOREIGN KEY (session_id) REFERENCES ShoppingSession(session_id)
);

-- Tabla de items comprados
CREATE TABLE IF NOT EXISTS PurchaseItem (
    purchase_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    recommendation_id INTEGER,
    promotion_id INTEGER,
    unit_price REAL NOT NULL,
    discount REAL DEFAULT 0.0,
    FOREIGN KEY (purchase_id) REFERENCES Purchase(purchase_id),
    FOREIGN KEY (item_id) REFERENCES Item(item_id),
    FOREIGN KEY (recommendation_id) REFERENCES Recommendation(recommendation_id),
    FOREIGN KEY (promotion_id) REFERENCES Promotion(promotion_id),
    FOREIGN KEY (customer_id) REFERENCES CustomerInfo(customer_id)
);

-- Tabla de eventos de emoción
CREATE TABLE IF NOT EXISTS EmotionEvent (
    emotion_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    dominant_label TEXT,
    group_label TEXT,
    emotions TEXT,
    avg_window REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES CustomerInfo(customer_id),
    FOREIGN KEY (session_id) REFERENCES ShoppingSession(session_id)
);

-- Tabla de posiciones de bolsas (para heatmaps)
CREATE TABLE IF NOT EXISTS BagPosition (
    position_id INTEGER PRIMARY KEY AUTOINCREMENT,
    bag_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES CustomerInfo(customer_id),
    FOREIGN KEY (session_id) REFERENCES ShoppingSession(session_id)
);

-- Tabla de recomendaciones generadas
CREATE TABLE IF NOT EXISTS Recommendation (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    weight REAL NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES CustomerInfo(customer_id),
    FOREIGN KEY (product_id) REFERENCES Product(product_id)
);

-- Tabla de promociones activas
CREATE TABLE IF NOT EXISTS Promotion (
    promotion_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    discount_percentage REAL NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES CustomerInfo(customer_id),
    FOREIGN KEY (product_id) REFERENCES Product(product_id)
);

-- Tabla de alertas activas
CREATE TABLE IF NOT EXISTS AlertEvent (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    emotion TEXT,
    session_id TEXT NOT NULL,
    location TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de alertas resueltas
CREATE TABLE IF NOT EXISTS AlertResolution (
    resolution_id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id INTEGER NOT NULL,
    staff_id TEXT NOT NULL,
    status TEXT NOT NULL,
    resolution_notes TEXT,
    resolved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índices para mejorar rendimiento
CREATE INDEX IF NOT EXISTS idx_purchase_customer ON Purchase(customer_id);
CREATE INDEX IF NOT EXISTS idx_recommendation_customer ON Recommendation(customer_id);
CREATE INDEX IF NOT EXISTS idx_emotion_customer ON EmotionEvent(customer_id);
CREATE INDEX IF NOT EXISTS idx_session_customer ON ShoppingSession(customer_id);
CREATE INDEX IF NOT EXISTS idx_product_popularity ON Product(popularity DESC);
CREATE INDEX IF NOT EXISTS idx_bag_movement ON BagPosition(customer_id, session_id, timestamp);