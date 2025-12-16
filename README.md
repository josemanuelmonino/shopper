# 🛍️ SHOPPER

### Smart Heatmaps, Offers, Personalization & Purchase Experience in Retail

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Computer Vision](https://img.shields.io/badge/AI-SFace%20%7C%20DeepFace-green)
![Machine Learning](https://img.shields.io/badge/LightGBM%20%7C%20MiniBatchKMeans-yellow)
![Status](https://img.shields.io/badge/Status-Prototype-orange)

**SHOPPER** is a *Smart Retail* platform designed to enhance the in-store shopping experience through **Computer Vision**, **Artificial Intelligence**, and **Data Analytics**. The system enables real-time personalization, dynamic pricing, emotional analysis of customers, and advanced visualization of in-store behavior, effectively bridging the gap between physical and digital retail.

This repository contains the full implementation of the system, including customer-facing applications (Smart Mirror), management dashboards, and simulation and analytics engines.

---

## 📑 Table of Contents

* [Key Features](#-key-features)
* [Project Architecture](#-project-architecture)
* [Technologies Used](#-technologies-used)
* [Installation & Setup](#-installation--setup)
* [Execution](#-execution)
* [Simulation & Data](#-simulation--data)
* [Authors](#-authors)

---

## ✨ Key Features

### 🪞 Customer Experience – Smart Mirror

An interactive system simulating a smart mirror inside the store:

* **Facial Recognition:** identification of registered customers using facial embeddings with **SFace** and **DeepFace**.
* **Emotion AI:** real-time detection of emotional states (happiness, surprise, frustration). When frustration is detected, alerts can be triggered for store staff.
* **Personalized Recommendations:** hybrid recommendation engine based on purchase history, customer clustering, and association rules (Apriori).

### 📊 Business Intelligence – Manager Dashboard

Web-based dashboard for strategic decision-making:

* **Dynamic Pricing:** automatic price adjustment based on demand estimation and business rules using *Machine Learning* models (**LightGBM**).
* **Heatmaps:** visualization of high-traffic store areas based on simulated customer trajectories.
* **Real-Time KPIs:** monitoring of key metrics such as total sales, customer satisfaction, and inventory status.

---

## 🏗️ Project Overview

The codebase follows a modular structure, clearly separating user interfaces, business logic, and data layers:

```text
SHOPPER/
├── app_customer/           # Customer Interface (Smart Mirror)
│   └── shopper_app.py
├── app_manager/            # Management Dashboard
│   └── manager_app.py
├── data/                   # Data and trained models
│   ├── facialrecognition/
│   ├── models/
│   └── raw/
├── database/               # Persistence layer
│   ├── schema.sql
│   └── shopper.db
├── src/                    # Core system logic
│   ├── ml_analytics/       # Emotion AI, pricing and heatmaps
│   ├── simulation/         # Customer and sales simulators
│   └── setup_db.py
├── tests/                  # Unit tests
└── environment.yml         # Conda environment
```

---

## 🛠️ Technologies Used

**Language:** Python 3.10+

**Web Framework:** Streamlit

**Computer Vision:**

* YOLOv8-Face – real-time face detection
* DeepFace / SFace – facial embeddings and emotion analysis

**Machine Learning:**

* LightGBM – dynamic pricing engine
* Scikit-learn – clustering and recommendation systems

**Database:** SQLite (local prototyping)

**Dependency Management:** Conda (optional – you can use any environment manager)

---

## ⚙️ Installation & Setup

1. **Clone the repository**

```bash
git clone <REPOSITORY_URL>
cd SHOPPER
```

2. **Create the virtual environment**

```bash
conda env create -f environment.yml
conda activate shopper
```

(Optional) **You may initialize the database, although it is already created.**

```bash
python src/setup_db.py
```

This script creates the database schema and loads synthetic data for simulation.

---

## ▶️ Execution

It is recommended to use two terminals:

**Terminal 1 – Smart Mirror (Customer App)**

```bash
streamlit run app_customer/shopper_app.py
```

**Terminal 2 – Manager Dashboard**

```bash
streamlit run app_manager/manager_app.py
```

---

## 🔄 Simulation & Data

The project includes simulation modules to generate realistic retail scenarios:

* **Synthetic customers:** generated profiles with preferences and behavior patterns.
* **Sales simulation:** transaction history used to train recommendation models.
* **Customer trajectories:** movement data used for heatmap generation.

---

## 👥 Authors (G1 GROUP - ETSIST UPM PIDS) 

* Lucía Pintos López
* Cristina Inés García-Yáñez
* José Manuel Moñino Glez
* Miguel Ángel Conde Ramírez
