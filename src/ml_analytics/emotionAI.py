from deepface import DeepFace
import cv2
from collections import deque
import time
import matplotlib.pyplot as plt
import csv

class EmotionDetector:
    def __init__(self, camera_index=0, history_len=300, frame_skip=15):
        """
        Inicializa el detector de emociones.
        :param camera_index: Índice de la cámara (0 por defecto).
        :param history_len: Tamaño del buffer de historial.
        :param frame_skip: Cada cuántos frames se ejecuta DeepFace.
        """
        self.camera_index = camera_index
        self.frame_skip = frame_skip
        self.cap = None
        
        # Configuración de grupos
        self.EMOTION_GROUPS = {
            "frustration": ["angry", "sad", "fear", "disgust"],
            "surprise": ["surprise"],
            "happiness": ["happy"],
        }

        # Configuración de picos
        self.PEAK_CONFIG = {
            "frustration": {"threshold": 60.0, "delta": 8, "window": 8.0, "cooldown": 30.0},
            "surprise":    {"threshold": 60.0, "delta": 8, "window": 4.0, "cooldown": 15.0},
            "happiness":   {"threshold": 60.0, "delta": 8, "window": 8.0, "cooldown": 30.0},
        }

        # Estado del sistema
        self.emotion_history = deque(maxlen=history_len)
        self.last_alert_time = {k: 0.0 for k in self.EMOTION_GROUPS.keys()}
        self.streak_counter = {k: 0 for k in self.EMOTION_GROUPS.keys()}
        self.detected_peaks = []
        
        # Variables de visualización
        self.ultima_emocion = None
        self.ultima_region = None

    def _get_group_score(self, group_name, emotions_dict):
        """Suma los porcentajes de las emociones que componen un grupo."""
        emos = self.EMOTION_GROUPS[group_name]
        return sum(float(emotions_dict.get(e, 0.0)) for e in emos)

    def _update_emotion_history(self, emotions_dict):
        """Guarda la emoción actual con timestamp."""
        self.emotion_history.append({
            "t": time.time(),
            "emotions": dict(emotions_dict)
        })

    def _compute_stats_for_group(self, group_name, now, window_seconds):
        """Devuelve (score_actual, media_en_ventana)."""
        values = []
        for item in self.emotion_history:
            if now - item["t"] <= window_seconds:
                values.append(self._get_group_score(group_name, item["emotions"]))

        if not values:
            return 0.0, 0.0

        current = values[-1]
        avg = sum(values) / len(values)
        return current, avg

    def _detect_peaks(self):
        """Revisa el historial y detecta alertas basadas en la configuración."""
        now = time.time()
        
        for group_name, cfg in self.PEAK_CONFIG.items():
            current, avg = self._compute_stats_for_group(group_name, now, cfg["window"])

            # 1. Comprobar umbral
            if current >= cfg["threshold"]:
                self.streak_counter[group_name] += 1
            else:
                self.streak_counter[group_name] = 0

            # 2. Comprobar racha y cooldown
            if (self.streak_counter[group_name] >= cfg["delta"] and 
                (now - self.last_alert_time[group_name]) >= cfg["cooldown"]):
                
                self.last_alert_time[group_name] = now
                
                # Guardar pico detectado
                self.detected_peaks.append({
                    "timestamp": now,
                    "emotion": group_name,
                    "value": current,
                    "avg_window": avg
                })
                
                # Resetear racha tras alerta
                self.streak_counter[group_name] = 0

    def _draw_overlay(self, frame):
        """Dibuja rectángulos y texto sobre el frame."""
        if self.ultima_emocion is None:
            return

        emociones_ordenadas = dict(sorted(self.ultima_emocion.items(), key=lambda x: x[1], reverse=True))
        region = self.ultima_region

        if region is not None:
            x, y = int(region.get('x', 0)), int(region.get('y', 0))
            w, h = int(region.get('w', 0)), int(region.get('h', 0))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text_x = x + w + 20
            text_y = y - 20
            if text_y < 20: text_y = 20

            for emo, pct in emociones_ordenadas.items():
                texto = f"{emo}: {pct:.1f}%"
                (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Fondo negro para texto
                cv2.rectangle(frame, (text_x - 5, text_y - th - 5), (text_x + tw + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(frame, texto, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                text_y += 18
        else:
            # Fallback si no hay región
            x0, y0 = 30, 40
            for emo, pct in emociones_ordenadas.items():
                texto = f"{emo}: {pct:.1f}%"
                (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x0 - 5, y0 - th - 5), (x0 + tw + 5, y0 + 5), (0, 0, 0), -1)
                cv2.putText(frame, texto, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                y0 += 20

    def run(self):
        """Inicia el bucle principal de captura y análisis."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("[ERROR] No se pudo abrir la cámara")
            return

        print("[INFO] Cámara abierta. Presiona 'q' para salir.")
        frame_count = 0

        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("[ERROR] No se pudo leer frame")
                    break

                frame_count += 1

                # Análisis periódico
                if frame_count % self.frame_skip == 0:
                    try:
                        analisis_rt = DeepFace.analyze(
                            img_path=frame,
                            actions=['emotion'],
                            enforce_detection=False,
                            silent=True # Para evitar spam en consola
                        )
                        
                        resultado = analisis_rt[0] if isinstance(analisis_rt, list) else analisis_rt
                        
                        self.ultima_emocion = resultado['emotion']
                        self.ultima_region = resultado.get('region') or resultado.get('box')

                        self._update_emotion_history(self.ultima_emocion)
                        self._detect_peaks()

                    except Exception as e:
                        # Si falla DeepFace, mantenemos el último estado
                        pass

                self._draw_overlay(frame)
                cv2.imshow("Emotion AI - DeepFace (Clase)", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self._save_results()

    def _save_results(self):
        """Genera gráficas y guarda CSV al finalizar."""
        print("\n[INFO] Generando reportes...")
        
        # 1. Graficar
        if len(self.emotion_history) > 0:
            t0 = self.emotion_history[0]["t"]
            times = [item["t"] - t0 for item in self.emotion_history]

            plt.figure(figsize=(10, 5))
            
            colors = {"frustration": "red", "surprise": "blue", "happiness": "green"}
            
            for group in self.EMOTION_GROUPS.keys():
                vals = [self._get_group_score(group, item["emotions"]) for item in self.emotion_history]
                plt.plot(times, vals, label=group.capitalize(), color=colors.get(group, "gray"))

            plt.xlabel("Tiempo (s)")
            plt.ylabel("Porcentaje (%)")
            plt.title("Evolución de emociones en el tiempo")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("grafica_emociones.png")
            print(" -> Gráfica guardada: grafica_emociones.png")
            plt.show()
        else:
            print(" -> No hay datos suficientes para graficar.")

        # 2. Guardar CSV
        if self.detected_peaks:
            filename = "peaks.csv"
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "emotion", "value", "avg_window"])
                for p in self.detected_peaks:
                    writer.writerow([p["timestamp"], p["emotion"], p["value"], p["avg_window"]])
            print(f" -> CSV guardado: {filename}")
        else:
            print(" -> No hubo picos detectados para guardar en CSV.")

# --- Bloque de ejecución ---
if __name__ == "__main__":
    detector = EmotionDetector(camera_index=0, frame_skip=15)
    detector.run()