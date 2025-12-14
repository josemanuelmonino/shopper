import os
import cv2
import time
import queue
import av
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from deepface import DeepFace
from streamlit_webrtc import VideoTransformerBase
import json
from src.db_manager import DataManagerSimple
import pandas as pd

class FaceEmotionProcessor(VideoTransformerBase):
    def __init__(self):
        # ---------------------------------------------------------
        # 1. CONFIGURACIÓN E INICIALIZACIÓN
        # ---------------------------------------------------------
        self.db_path = os.path.abspath("data/facialrecognition")
        self.yolo_path = os.path.abspath("src/ml_analytics/yolov8n-face.pt")
        
        self._setup_directories()

        # Cargar YOLO
        try:
            print(f"[INFO] Cargando YOLO desde: {self.yolo_path}")
            self.model = YOLO(self.yolo_path)
            self.model_loaded = True
        except Exception as e:
            print(f"[ERROR] Fallo cargando YOLO: {e}")
            self.model_loaded = False

        # --- EXECUTOR (Igual que tu script funcional) ---
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending_future = None

        # ---------------------------------------------------------
        # 2. VARIABLES DE ESTADO
        # ---------------------------------------------------------
        # Configuración de captura
        self.PHASES = ["frontal", "izquierda", "derecha"]
        self.CAPTURE_CONFIG = {
            "frontal":   {"frames": 30, "keep": 2, "msg": "Mira al frente"},
            "izquierda": {"frames": 15, "keep": 1, "msg": "Mira ligeramente hacia la izquierda"},
            "derecha":   {"frames": 15, "keep": 1, "msg": "Mira ligeramente hacia la derecha"},
        }
        self.SHARPNESS_MIN = 60.0 

        # Estado de Captura
        self.capture_mode = False        
        self.capture_phase_idx = 0       
        self.capture_buffer = []         
        self.capture_started = False     
        self.quiet_cap_until = 0.0       
        self.frozen_person_name = None   
        self.phase_completed = False     

        # Estado de Reconocimiento
        self.RECOG_EVERY = 30 
        self.frame_count = 0
        self.last_face_name = "Unknown"
        
        # UI "Estate quieto"
        self.quiet_rec_until = 0.0
        self.rec_msg_started = False

        # Cola para Streamlit (Auto-login)
        self.result_queue = queue.Queue()

        # Para qué no esté constantemente buscando logins 
        self.last_enqueued = None
        self.last_enqueued_ts = 0.0
        self.LOCK_AFTER_ENQUEUE = False
        self.locked_identity = None
        self.recognition_enabled = True



    def _setup_directories(self):
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

    # ==========================================================
    # MÉTODOS DE UTILIDAD (Basados en tu script)
    # ==========================================================
    def _frame_sharpness(self, gray):
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _put_multiline_text(self, img, text, x, y, max_width, line_height=30, scale=0.8, color=(0,255,255)):
        """Tu función para escribir texto que salta de línea"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        words = text.split(" ")
        line = ""
        lines = []

        for w in words:
            test_line = line + w + " "
            (tw, th), _ = cv2.getTextSize(test_line, font, scale, 2)
            if tw > max_width:
                lines.append(line)
                line = w + " "
            else:
                line = test_line
        lines.append(line)

        for i, l in enumerate(lines):
            cv2.putText(img, l, (x, y + i*line_height), font, scale, color, 2, cv2.LINE_AA)

    # ==========================================================
    # WORKER DE RECONOCIMIENTO (Ejecutado por el Executor)
    # ==========================================================
    def _recognize_task(self, face_rgb):
        """Tarea pesada que corre en segundo plano"""
        try:
            # DeepFace con detector_backend="skip" para evitar error de OpenCV
            dfs = DeepFace.find(
                img_path=face_rgb, 
                db_path=self.db_path, 
                model_name="SFace", 
                detector_backend="skip",
                distance_metric="cosine", 
                enforce_detection=False,
                silent=True
            )
            
            detected_name = "Unknown"
            if isinstance(dfs, list) and len(dfs) > 0 and not dfs[0].empty:
                identity_path = dfs[0].iloc[0]["identity"]
                detected_name = os.path.basename(os.path.dirname(identity_path))
                print(f"[DEBUG] Encontrado: {detected_name}")

            return detected_name
            
        except Exception as e:
            print(f"[WARN] Error DeepFace: {e}")
            return "Unknown"

    # ==========================================================
    # LÓGICA DE CAPTURA (REGISTRO)
    # ==========================================================
    def start_capture_sequence(self, name):
        self.frozen_person_name = name
        self.capture_mode = True
        self.capture_phase_idx = 0
        self.capture_buffer = []
        self.phase_completed = False
        self.quiet_cap_until = time.time() + 3.0 
        print(f"[INFO] Iniciando captura para: {name}")

    def continue_next_phase(self):
        if self.phase_completed:
            self.capture_phase_idx += 1
            if self.capture_phase_idx < len(self.PHASES):
                self.phase_completed = False
                self.capture_buffer = []
                self.quiet_cap_until = time.time() + 3.0
                self.capture_started = False
            else:
                self.capture_mode = False
                print("[INFO] Registro finalizado")

    def _handle_capture_process(self, frame, frame_vis, bbox):
        if self.phase_completed:
            cv2.putText(frame_vis, "Fase lista. Pulsa 'Continuar' ->", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return

        current_phase = self.PHASES[self.capture_phase_idx]
        cfg = self.CAPTURE_CONFIG[current_phase]
        x, y, w, h = bbox
        h_frame, w_frame = frame_vis.shape[:2]

        # 1. Mensaje de Preparación
        if time.time() < self.quiet_cap_until:
            msg = f"{cfg['msg']} ..."
            self._put_multiline_text(frame_vis, msg, 10, 40, w_frame - 20)
            self.capture_started = True
            return

        # 2. Captura
        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size > 0:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            sharpness = self._frame_sharpness(gray)

            if sharpness >= self.SHARPNESS_MIN:
                self.capture_buffer.append((sharpness, frame.copy()))
                cv2.putText(frame_vis, f"Capturando {current_phase}: {len(self.capture_buffer)}/{cfg['frames']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 3. Guardado
        if len(self.capture_buffer) >= cfg["frames"]:
            # Guardar fotos
            self.capture_buffer.sort(key=lambda x: x[0], reverse=True)
            best = self.capture_buffer[:cfg["keep"]]
            
            save_dir = os.path.join(self.db_path, self.frozen_person_name)
            os.makedirs(save_dir, exist_ok=True)

            for i, (sharp, img) in enumerate(best):
                filename = f"{current_phase}_{int(sharp)}_{i}.jpg"
                path = os.path.join(save_dir, filename)
                cv2.imwrite(path, img)
            
            self.phase_completed = True

    # ==========================================================
    # BUCLE PRINCIPAL (recv)
    # ==========================================================
    def recv(self, frame):
        # 1. Convertir formato AV -> OpenCV (BGR)
        img = frame.to_ndarray(format="bgr24")
        
        if not self.model_loaded:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        self.frame_count += 1
        frame_vis = img.copy()
        h_vis, w_vis = frame_vis.shape[:2]

        # 2. Detección YOLO
        results = self.model(img, verbose=False)
        bbox = None
        if len(results[0].boxes) > 0:
            # Obtener la caja más grande
            box = max(results[0].boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # -----------------------------------------------------
            # AÑADIR PADDING (MARGEN) AQUÍ
            # -----------------------------------------------------
            h_img, w_img = img.shape[:2]
            w_box = x2 - x1
            h_box = y2 - y1
            
            # Calculamos un 20% de margen
            pad_x = int(w_box * 0.20)
            pad_y = int(h_box * 0.20)

            # Expandimos coordenadas sin salirnos de la imagen
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w_img, x2 + pad_x)
            y2 = min(h_img, y2 + pad_y)
            # -----------------------------------------------------

            bbox = (x1, y1, x2-x1, y2-y1)
        # 3. Lógica según estado
        if bbox:
            x, y, w, h = bbox

            # --- A. MODO REGISTRO (Captura) ---
            if self.capture_mode and self.capture_phase_idx < len(self.PHASES):
                cv2.rectangle(frame_vis, (x, y), (x+w, y+h), (255, 165, 0), 2)
                self._handle_capture_process(img, frame_vis, bbox)

            else:
                # Dibujar rectángulo
                color = (0, 255, 0) if self.last_face_name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame_vis, (x, y), (x+w, y+h), color, 2)

                # ✅ RECONOCIMIENTO SOLO EN MODO IDENTIFICACIÓN
                if (self.locked_identity is None) and self.recognition_enabled:
                    if self.frame_count % self.RECOG_EVERY == 0:
                        if self.pending_future is None or self.pending_future.done():
                            face_crop = img[y:y+h, x:x+w]
                            if face_crop.size > 0:
                                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                                self.pending_future = self.executor.submit(self._recognize_task, face_rgb)

                    # Comprobar si hay resultados listos del hilo
                    if self.pending_future is not None and self.pending_future.done():
                        try:
                            detected_name = self.pending_future.result()
                            self.last_face_name = detected_name

                            if detected_name != "Unknown":
                                now = time.time()

                                TIME_BETWEEN_SAME_USER = 0.5 

                                if (detected_name != self.last_enqueued) or \
                                   ((now - self.last_enqueued_ts) > TIME_BETWEEN_SAME_USER):
                                    
                                    self.result_queue.put(detected_name)
                                    self.last_enqueued = detected_name
                                    self.last_enqueued_ts = now
                                    print(f"[DEBUG] Encolado para login: {detected_name}")

                            self.pending_future = None
                        except Exception as e:
                            print(f"[ERROR] En thread: {e}")

                # UI: Mensaje "Estate quieto"
                if not self.rec_msg_started:
                    self.quiet_rec_until = time.time() + 3.0
                    self.rec_msg_started = True

                if time.time() < self.quiet_rec_until:
                    self._put_multiline_text(frame_vis, "Estate quieto, reconociendo...", 10, 40, w_vis - 20)

                # UI: Nombre
                cv2.putText(
                    frame_vis,
                    f"ID: {self.last_face_name}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        else:
            # Si no hay cara, reseteamos el mensaje de "Estate quieto" para la próxima vez
            self.rec_msg_started = False

        return av.VideoFrame.from_ndarray(frame_vis, format="bgr24")
    
# ==========================================================
# PROCESADOR DE EMOCIONES EN VIVO (POST-LOGIN)
# ==========================================================
import os
import cv2
import time
import json
import av
import pandas as pd
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from deepface import DeepFace
from streamlit_webrtc import VideoTransformerBase

# Tu DataManager
from db_manager import DataManagerSimple 

class EmotionLiveProcessor(VideoTransformerBase):
    def __init__(self):
        # 1. INICIALIZACIÓN
        self.customer_id = "Unknown" 
        self.session_id = "NoSession"
        
        # --- CAMBIO IMPORTANTE ---
        # No instanciamos la DB aquí para evitar error de "Objects created in a thread..."
        # Solo guardamos la ruta.
        self.db_path = "database/shopper.db"

        # 2. CONFIGURACIÓN
        self.EMOTION_GROUPS = {
            "frustration": ["angry", "sad", "fear", "disgust"],
            "surprise":    ["surprise"],
            "happiness":   ["happy"],
        }
        
        self.PEAK_CONFIG = {
            "frustration": {"threshold": 50.0, "delta": 4, "window": 4.0, "cooldown": 10.0},
            "surprise":    {"threshold": 60.0, "delta": 3, "window": 2.0, "cooldown": 5.0},
            "happiness":   {"threshold": 60.0, "delta": 5, "window": 5.0, "cooldown": 15.0},
        }

        # 3. ESTADO
        self.history_len = 100
        self.emotion_history = deque(maxlen=self.history_len)
        self.last_alert_time = {k: 0.0 for k in self.EMOTION_GROUPS.keys()}
        self.streak_counter = {k: 0 for k in self.EMOTION_GROUPS.keys()}
        
        # Threading y Visuales
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.last_emotions = None 
        self.last_region = None
        self.frame_skip_counter = 0
        self.FRAME_SKIP = 10 

    def update_user_info(self, customer_id, session_id):
        self.customer_id = customer_id
        self.session_id = session_id

    # ---------------------------------------------------------
    # LÓGICA DE NEGOCIO
    # ---------------------------------------------------------
    def _get_group_score(self, group_name, emotions_dict):
        emos = self.EMOTION_GROUPS[group_name]
        return sum(float(emotions_dict.get(e, 0.0)) for e in emos)

    def _compute_stats(self, group_name, now, window_seconds):
        values = []
        for item in self.emotion_history:
            if now - item["t"] <= window_seconds:
                values.append(self._get_group_score(group_name, item["emotions"]))
        if not values: return 0.0, 0.0
        return values[-1], sum(values) / len(values)

    def _save_event_to_db(self, group_label, emotions_raw, avg_val):
        """Guarda en DB creando la conexión EN ESTE HILO"""
        if self.customer_id == "Unknown":
            return 

        try:
            # 1. Crear conexión LOCAL al hilo (Soluciona error SQLite thread)
            local_dm = DataManagerSimple(self.db_path)

            # 2. Convertir float32 a float (Soluciona error JSON)
            emotions_serializable = {k: float(v) for k, v in emotions_raw.items()}
            dominant_label = max(emotions_serializable, key=emotions_serializable.get)
            
            event_df = pd.DataFrame([{
                "customer_id": self.customer_id,
                "session_id": self.session_id,
                "dominant_label": dominant_label,
                "group_label": group_label,
                "emotions": json.dumps(emotions_serializable),
                "avg_window": round(float(avg_val), 2),
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }])
            
            local_dm.save_df(event_df, "EmotionEvent", if_exists="append")
            print(f"[DB] Evento guardado: {group_label}")
            
        except Exception as e:
            print(f"[ERROR DB] {e}")

    def _process_deepface(self, frame_rgb):
        try:
            analysis = DeepFace.analyze(
                img_path=frame_rgb, actions=['emotion'],
                enforce_detection=False, detector_backend="opencv", silent=True
            )
            if isinstance(analysis, list): analysis = analysis[0]
            return analysis
        except Exception:
            return None

    def _update_logic(self, result):
        now = time.time()
        emotions = result['emotion']
        self.emotion_history.append({"t": now, "emotions": emotions})

        for group, cfg in self.PEAK_CONFIG.items():
            current, avg = self._compute_stats(group, now, cfg["window"])
            
            if current >= cfg["threshold"]:
                self.streak_counter[group] += 1
            else:
                self.streak_counter[group] = 0

            if (self.streak_counter[group] >= cfg["delta"] and 
                (now - self.last_alert_time[group]) >= cfg["cooldown"]):
                
                self.last_alert_time[group] = now
                self.streak_counter[group] = 0
                self._save_event_to_db(group, emotions, avg)

    # ---------------------------------------------------------
    # VISUALIZACIÓN (DRAW OVERLAY)
    # ---------------------------------------------------------
    def _draw_overlay(self, frame):
        """Dibuja rectángulos y texto sobre el frame."""
        if self.last_emotions is None:
            return

        # Ordenar emociones de mayor a menor
        emociones_ordenadas = dict(sorted(self.last_emotions.items(), key=lambda x: x[1], reverse=True))
        region = self.last_region

        if region is not None:
            # Obtener coordenadas de la caja
            x = int(region.get('x', 0))
            y = int(region.get('y', 0))
            w = int(region.get('w', 0))
            h = int(region.get('h', 0))
            
            # Asegurar que no sean negativas
            x, y = max(0, x), max(0, y)

            # Dibujar caja cara
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Coordenadas para el texto (a la derecha de la cara)
            text_x = x + w + 20
            text_y = y + 20 

            # Asegurar que el texto no se salga por arriba
            if text_y < 20: text_y = 20
            
            # Limite derecho de la pantalla
            frame_w = frame.shape[1]

            for emo, pct in emociones_ordenadas.items():
                texto = f"{emo}: {pct:.1f}%"
                (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Ajuste si se sale por la derecha
                draw_x = text_x
                if draw_x + tw > frame_w:
                    draw_x = x - tw - 10 

                # Fondo negro para texto
                cv2.rectangle(frame, (draw_x - 5, text_y - th - 5), (draw_x + tw + 5, text_y + 5), (0, 0, 0), -1)
                # Texto verde
                cv2.putText(frame, texto, (draw_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                text_y += 20 
        else:
            # Fallback si no hay región detectada
            x0, y0 = 30, 40
            for emo, pct in emociones_ordenadas.items():
                texto = f"{emo}: {pct:.1f}%"
                (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                cv2.rectangle(frame, (x0 - 5, y0 - th - 5), (x0 + tw + 5, y0 + 5), (0, 0, 0), -1)
                cv2.putText(frame, texto, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                y0 += 25

    # ---------------------------------------------------------
    # BUCLE PRINCIPAL
    # ---------------------------------------------------------
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. THREADING IA
        self.frame_skip_counter += 1
        if self.frame_skip_counter % self.FRAME_SKIP == 0:
            if self.future and self.future.done():
                try:
                    res = self.future.result()
                    if res:
                        self.last_emotions = res['emotion']
                        self.last_region = res.get('region') or res.get('box')
                        self._update_logic(res)
                except Exception:
                    pass
            
            if self.future is None or self.future.done():
                frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.future = self.executor.submit(self._process_deepface, frame_rgb)

        # 2. DIBUJAR VISUALIZACIÓN
        frame_vis = img.copy()
        self._draw_overlay(frame_vis)

        return av.VideoFrame.from_ndarray(frame_vis, format="bgr24")