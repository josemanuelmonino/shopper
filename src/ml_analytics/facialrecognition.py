import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor

class FaceRecognitionSystem:
    def __init__(self, db_path="db_shopper", camera_index=0):
        """
        Inicializa el sistema de reconocimiento y captura facial.
        """
        self.db_path = db_path
        self.camera_index = camera_index
        
        # Configuración inicial
        self._setup_directories()
        
        # Carga de modelos
        print("[INFO] Cargando modelo YOLOv8-face...")
        self.yolo_model = YOLO(r"C:\Users\mikic\OneDrive - Universidad Politécnica de Madrid\ProyectosDatos\shopper\src\ml_analytics\yolov8n-face.pt")
        
        # Threading para reconocimiento
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending_future = None
        
        # Configuración de parámetros
        self.RECOG_EVERY = 60          # Frames entre reconocimientos
        self.SHARPNESS_MIN = 90.0      # Umbral de nitidez
        
        # Configuración de fases de captura
        self.PHASES = ["frontal", "izquierda", "derecha"]
        self.CAPTURE_CONFIG = {
            "frontal":   {"frames": 30, "keep": 2, "msg": "Mira al frente"},
            "izquierda": {"frames": 15, "keep": 1, "msg": "Mira ligeramente hacia la izquierda"},
            "derecha":   {"frames": 15, "keep": 1, "msg": "Mira ligeramente hacia la derecha"},
        }

        # Estado del sistema
        self.frame_count = 0
        self.last_results = []         # [(name, distance, bbox), ...]
        self.frozen_person_name = None # Nombre fijo durante la sesión de captura
        self.modo_identificacion = False
        
        # Variables de UI
        self.quiet_rec_until = 0.0
        self.rec_msg_started = False
        
        # Variables de control de Captura
        self.capture_mode = False
        self.capture_phase_idx = 0     # Índice en self.PHASES
        self.capture_buffer = []       # [(sharpness, frame), ...]
        self.capture_started = False   # True tras el tiempo de preparación
        self.quiet_cap_until = 0.0     # Timestamp para fin de mensaje de preparación

    def _setup_directories(self):
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

    # -------------------------------------------------------------------
    # MÉTODOS AUXILIARES (Estáticos / Internos)
    # -------------------------------------------------------------------

    @staticmethod
    def _frame_sharpness(gray):
        """Calcula la varianza del Laplaciano como medida de nitidez."""
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def _put_multiline_text(img, text, x, y, max_width, line_height=30, scale=0.8, color=(0,255,255)):
        """Dibuja texto multilínea."""
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

    # -------------------------------------------------------------------
    # LÓGICA DE VISIÓN (Detect & Recognize)
    # -------------------------------------------------------------------

    def _detect_biggest_face(self, frame):
        """Usa YOLO para detectar la cara más grande."""
        results = self.yolo_model(frame, verbose=False)
        if len(results[0].boxes) == 0:
            return None

        # Buscar la caja con mayor área
        box = max(
            results[0].boxes,
            key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
        )
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        return (x1, y1, x2 - x1, y2 - y1) # x, y, w, h

    def _recognize_worker(self, frame_rgb):
        """Función que se ejecuta en el hilo secundario."""
        try:
            dfs = DeepFace.find(
                img_path=frame_rgb,
                db_path=self.db_path,
                model_name="SFace",
                distance_metric="cosine",
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(dfs, list):
                df = dfs[0]
            else:
                df = dfs

            if df is None or df.empty:
                return []

            best_match = df.iloc[0]
            identity_path = best_match["identity"]
            distance = best_match["distance"]
            # Extraer nombre de la carpeta padre
            person_name = os.path.basename(os.path.dirname(identity_path))

            return [(person_name, distance, None)] # bbox no es necesario aquí
        except Exception as e:
            print(f"[WARN] DeepFace error: {e}")
            return []

    def _manage_recognition_thread(self, frame, bbox):
        """Gestiona el hilo de reconocimiento en segundo plano."""
        if not self.modo_identificacion or bbox is None:
            return

        x, y, w, h = bbox
        
        # Lanzar hilo cada X frames si no hay uno corriendo
        if self.frame_count % self.RECOG_EVERY == 0:
            if self.pending_future is None or self.pending_future.done():
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size > 0:
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    self.pending_future = self.executor.submit(self._recognize_worker, face_rgb.copy())

        # Recoger resultados si el hilo terminó
        if self.pending_future is not None and self.pending_future.done():
            try:
                res = self.pending_future.result()
                if res:
                    self.last_results = res
            except Exception as e:
                print(f"[ERROR] Hilo fallido: {e}")
            self.pending_future = None

    # -------------------------------------------------------------------
    # MÁQUINA DE ESTADOS: CAPTURA
    # -------------------------------------------------------------------

    def _handle_capture_process(self, frame, frame_vis, bbox):
        """Lógica principal de las 3 fases de captura."""
        if not self.capture_mode or bbox is None:
            return

        # Si ya terminamos todas las fases
        if self.capture_phase_idx >= len(self.PHASES):
            self.capture_mode = False
            print("[INFO] Proceso de captura completado.")
            return

        current_phase = self.PHASES[self.capture_phase_idx]
        cfg = self.CAPTURE_CONFIG[current_phase]
        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]

        # 1. Periodo de preparación (mensaje en pantalla)
        if time.time() < self.quiet_cap_until:
            msg = f"{cfg['msg']} ..."
            self._put_multiline_text(frame_vis, msg, 10, 40, w_frame - 20)
            return
        else:
            self.capture_started = True

        # 2. Captura activa
        if self.capture_started:
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size > 0:
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                sharpness = self._frame_sharpness(gray)

                if sharpness >= self.SHARPNESS_MIN:
                    self.capture_buffer.append((sharpness, frame.copy()))
                    
                    cv2.putText(frame_vis, f"Capturando {current_phase}: {len(self.capture_buffer)}/{cfg['frames']}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 3. Verificar fin de fase
        if len(self.capture_buffer) >= cfg["frames"]:
            print(f"[INFO] Fase '{current_phase}' terminada.")
            self._save_best_captures(current_phase, cfg["keep"])
            
            # Avanzar a siguiente fase
            self.capture_buffer = []
            self.capture_started = False
            self.capture_phase_idx += 1
            
            if self.capture_phase_idx < len(self.PHASES):
                next_p = self.PHASES[self.capture_phase_idx]
                print(f"[INFO] Siguiente fase: {next_p}. Pulsa 'c' para continuar.")
                self.capture_mode = False # Pausamos hasta que usuario pulse 'c' de nuevo

    def _save_best_captures(self, phase, keep_n):
        """Guarda las fotos con mejor nitidez."""
        # Ordenar por nitidez descendente
        self.capture_buffer.sort(key=lambda x: x[0], reverse=True)
        best = self.capture_buffer[:keep_n]

        save_dir = os.path.join(self.db_path, self.frozen_person_name)
        os.makedirs(save_dir, exist_ok=True)

        for i, (sharp, img) in enumerate(best):
            filename = f"{phase}_{int(sharp)}_{i}.jpg"
            path = os.path.join(save_dir, filename)
            cv2.imwrite(path, img)
        
        print(f"[INFO] Guardadas {len(best)} imágenes en {save_dir}")

    # -------------------------------------------------------------------
    # UI Y VISUALIZACIÓN
    # -------------------------------------------------------------------

    def _draw_interface(self, frame_vis, bbox):
        h, w = frame_vis.shape[:2]

        if bbox is not None:
            x, y, wb, hb = bbox
            cv2.rectangle(frame_vis, (x, y), (x+wb, y+hb), (0, 255, 0), 2)

            # Nombre detectado
            if self.last_results:
                name = self.last_results[0][0]
                cv2.putText(frame_vis, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Mensaje inicial de "Reconocimiento en progreso"
        if self.modo_identificacion and not self.rec_msg_started and bbox is not None:
            self.quiet_rec_until = time.time() + 3.0
            self.rec_msg_started = True

        if time.time() < self.quiet_rec_until:
            self._put_multiline_text(frame_vis, "Estate quieto, reconociendo...", 10, 40, w - 20)

    # -------------------------------------------------------------------
    # EJECUCIÓN PRINCIPAL
    # -------------------------------------------------------------------

    def run(self):
        # 1. Configuración de Usuario
        print("\n--- SISTEMA DE GESTIÓN FACIAL ---")
        print("¿Ya tienes perfil de reconocimiento facial creado? (s/n)")
        resp = input("> ").strip().lower()
        
        if resp == 's':
            self.modo_identificacion = True
            print("[INFO] Modo IDENTIFICACIÓN activado.")
        else:
            self.modo_identificacion = False
            print("[INFO] Modo REGISTRO activado. Presiona 'c' cuando estés listo.")

        # 2. Bucle de Video
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("[ERROR] No se pudo abrir la cámara.")
            return

        print("[INFO] Cámara iniciada. 'c' para capturar/avanzar fase, 'q' para salir.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

                self.frame_count += 1
                frame_vis = frame.copy()
                
                # A. Detectar
                bbox = self._detect_biggest_face(frame)

                # B. Reconocer (Hilo)
                self._manage_recognition_thread(frame, bbox)

                # C. Dibujar UI básica
                self._draw_interface(frame_vis, bbox)

                # D. Gestionar Captura (si está activa)
                self._handle_capture_process(frame, frame_vis, bbox)

                cv2.imshow("Sistema Facial", frame_vis)
                
                # E. Controles
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                
                elif key == ord('c'):
                    # Iniciar o Continuar captura
                    if bbox is None:
                        print("[WARN] No se detecta cara para iniciar captura.")
                        continue
                    
                    if self.capture_phase_idx >= len(self.PHASES):
                        print("[INFO] Ya has completado todas las fases.")
                        continue

                    # Solo pedir nombre al inicio de la primera fase
                    if self.capture_phase_idx == 0 and self.frozen_person_name is None:
                        if self.modo_identificacion and self.last_results:
                            self.frozen_person_name = self.last_results[0][0]
                            print(f"[INFO] Usando nombre detectado: {self.frozen_person_name}")
                        else:
                            # Pausar visualización un momento para input de consola
                            cv2.destroyAllWindows() # Opcional, para dar foco a terminal
                            self.frozen_person_name = input("\n[INPUT] Escribe tu nombre para el nuevo perfil: ").strip()
                            cv2.namedWindow("Sistema Facial") # Recuperar ventana

                    if not self.capture_mode:
                        self.capture_mode = True
                        self.capture_buffer = []
                        self.quiet_cap_until = time.time() + 3.0 # 3 seg preparación
                        self.capture_started = False
                        print(f"[INFO] Iniciando fase: {self.PHASES[self.capture_phase_idx]}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.executor.shutdown(wait=False)
            print("[INFO] Sistema cerrado correctamente.")

# --- Bloque de ejecución ---
if __name__ == "__main__":
    # Asegúrate de estar en el directorio correcto o ajusta db_path
    # os.chdir(r"TU_RUTA_AQUI") 
    app = FaceRecognitionSystem(db_path=r"C:\Users\mikic\OneDrive - Universidad Politécnica de Madrid\ProyectosDatos\shopper\data\facialrecognition")
    app.run()