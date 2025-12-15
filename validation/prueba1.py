import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deepface import DeepFace
from sklearn.metrics import confusion_matrix, classification_report
import os

# 1. CONFIGURACIÓN
# Define tus grupos igual que en tu APP
EMOTION_GROUPS = {
    "frustration": ["angry", "sad", "fear", "disgust"],
    "surprise":    ["surprise"],
    "happiness":   ["happy"],
    "neutral":     ["neutral"] # Añadimos neutral por si acaso
}

base_dir = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(base_dir, "test_images")
LABELS = ["happiness", "surprise", "frustration", "neutral"]

def map_raw_to_group(raw_emotion):
    """Convierte la salida de DeepFace a tus grupos"""
    for group, emotions in EMOTION_GROUPS.items():
        if raw_emotion in emotions:
            return group
    return "unknown"

def run_validation():
    y_true = [] # La realidad (nombre de la carpeta)
    y_pred = [] # Lo que dice la IA

    print(f"🚀 Iniciando validación en '{TEST_DIR}'...\n")

    if not os.path.exists(TEST_DIR):
        print(f"[ERROR] No encuentro la carpeta '{TEST_DIR}'. Créala y pon fotos dentro.")
        return

    # Recorremos las carpetas
    for label in os.listdir(TEST_DIR):
        folder_path = os.path.join(TEST_DIR, label)
        
        if os.path.isdir(folder_path):
            if label not in LABELS:
                print(f"⚠️ Saltando carpeta '{label}' porque no es un grupo conocido.")
                continue

            print(f"📂 Procesando grupo: {label.upper()}")
            
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                
                try:
                    # Análisis con DeepFace (backend opencv para ir rápido)
                    obj = DeepFace.analyze(
                        img_path=img_path, 
                        actions=['emotion'],
                        detector_backend='opencv',
                        enforce_detection=False,
                        silent=True
                    )
                    
                    # DeepFace devuelve una lista, cogemos el primero
                    result = obj[0]
                    dominant_raw = result['dominant_emotion']
                    
                    # Mapeamos (ej: 'sad' -> 'frustration')
                    group_detected = map_raw_to_group(dominant_raw)
                    
                    # Guardamos resultados
                    y_true.append(label)
                    y_pred.append(group_detected)
                    
                    match_symbol = "✅" if label == group_detected else "❌"
                    print(f"   - {img_name}: Detectado '{dominant_raw}' -> Grupo '{group_detected}' {match_symbol}")

                except Exception as e:
                    print(f"   - [ERROR] {img_name}: {e}")

    # 3. GENERAR MATRIZ DE CONFUSIÓN
    print("\n📊 Generando gráficos...")
    
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    
    # Draw with Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=LABELS,
        yticklabels=LABELS
    )

    plt.xlabel('System Prediction')
    plt.ylabel('Ground Truth (Folder)')
    plt.title('Confusion Matrix: Emotion Validation')

    # Save the image
    plt.savefig("confusion_matrix_validation.png")
    print("✅ Plot saved as 'confusion_matrix_validation.png'")

    # 4. TEXT REPORT (Precision, Recall, F1)
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

    plt.show()


if __name__ == "__main__":
    run_validation()