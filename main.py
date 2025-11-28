import customtkinter as ctk
import cv2
import os
import sys
import threading
import pickle
import numpy as np
import face_recognition
import csv
import subprocess
import time
from datetime import datetime, timedelta
from PIL import Image, ImageTk
from ultralytics import YOLO

#SYSTEM CONFIG)
SAVE_DIR = 'data_guong_mat'
DATABASE_FILE = 'face_database.pkl'
LATE_THRESHOLD_MINUTES = 15 


SHIFTS = [
    {"name": "Ca 1", "start": "06:45", "end": "09:25"},
    {"name": "Ca 2", "start": "09:30", "end": "12:10"},
    {"name": "Ca 3", "start": "13:00", "end": "15:40"},
    {"name": "Ca 4", "start": "15:45", "end": "18:25"}
]


def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_base_path()
YOLO_MODEL_PATH = os.path.join(BASE_PATH, r'runs/detect/train4/weights/best.pt')


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# DIALOG

class AddStudentDialog(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Thêm Sinh Viên Mới")
        self.geometry("400x300")
        self.resizable(False, False)
        self.result = None
        self.attributes("-topmost", True) # Giữ cửa sổ luôn nổi

        # UI Components
        ctk.CTkLabel(self, text="Nhập thông tin sinh viên", font=("Arial", 18, "bold")).pack(pady=20)
        self.entry_name = ctk.CTkEntry(self, placeholder_text="Họ và Tên", width=300)
        self.entry_name.pack(pady=10)
        self.entry_msv = ctk.CTkEntry(self, placeholder_text="Mã Sinh Viên", width=300)
        self.entry_msv.pack(pady=10)
        ctk.CTkButton(self, text="Tiếp tục", command=self.on_submit, height=40).pack(pady=20)

    def on_submit(self):
        name = self.entry_name.get().strip()
        msv = self.entry_msv.get().strip()
        if name and msv:
            self.result = (name, msv)
            self.destroy()


# MAIN APP

class KioskApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # 1. Init Window Settings
        self.title("HỆ THỐNG ĐIỂM DANH SINH VIÊN")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height}")
        
        # 2. Grid Layout Configuration
        self.grid_columnconfigure(0, weight=3) # Camera Area
        self.grid_columnconfigure(1, weight=1) # Info Area
        self.grid_rowconfigure(0, weight=1)

        # 3. UI: Camera Frame (Left)
        self.video_frame = ctk.CTkFrame(self, fg_color="black")
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(fill="both", expand=True)

        # 4. UI: Info Panel (Right)
        self.info_frame = ctk.CTkFrame(self, fg_color="#2b2b2b", corner_radius=0)
        self.info_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        
        # Info Components
        self.lbl_shift_info = ctk.CTkLabel(self.info_frame, text="LOADING...", font=("Arial", 20, "bold"), text_color="#FFA500")
        self.lbl_shift_info.pack(pady=(30, 10))

        ctk.CTkLabel(self.info_frame, text="THÔNG TIN SINH VIÊN", font=("Arial", 24, "bold")).pack(pady=(10, 20))
        self.avatar_label = ctk.CTkLabel(self.info_frame, text="[No Image]", width=200, height=200, fg_color="#444")
        self.avatar_label.pack(pady=10)

        self.lbl_name = ctk.CTkLabel(self.info_frame, text="---", font=("Arial", 26, "bold"), text_color="white", wraplength=300)
        self.lbl_name.pack(pady=(20, 5))
        self.lbl_msv = ctk.CTkLabel(self.info_frame, text="MSV: ---", font=("Arial", 20), text_color="#AAAAAA")
        self.lbl_msv.pack(pady=5)
        self.lbl_time = ctk.CTkLabel(self.info_frame, text="--:--:--", font=("Arial", 50, "bold"), text_color="#00FFFF")
        self.lbl_time.pack(pady=10)
        self.lbl_status = ctk.CTkLabel(self.info_frame, text="SẴN SÀNG", font=("Arial", 22, "bold"), fg_color="#444", corner_radius=10, width=250, height=50)
        self.lbl_status.pack(pady=30)
        
        # Control Buttons (Hidden by default)
        self.btn_frame = ctk.CTkFrame(self.info_frame, fg_color="transparent")
        ctk.CTkButton(self.btn_frame, text="Thêm Sinh Viên", command=self.action_add_face).pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.btn_frame, text="Cập nhật Data", command=self.action_reload_db).pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.btn_frame, text="Mở Excel", command=self.action_open_excel, fg_color="green").pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.btn_frame, text="Thoát", command=self.close_app, fg_color="red").pack(fill="x", padx=20, pady=5)
        
        # Bind Events
        self.video_label.bind("<Button-1>", self.toggle_menu)
        self.info_frame.bind("<Button-1>", self.toggle_menu)
        self.menu_visible = False

        # 5. Initialize System
        self.is_running = True
        self.cap = None
        self.checked_in_today = set() 
        self.last_update_name = "" 
        self.last_update_time = 0

        self.load_resources()
        self.start_camera()

    #  RESOURCE MANAGEMENT 
    def load_resources(self):
        """Load Database & AI Models"""
        self.known_encodings = []
        self.known_ids = [] 
        if os.path.exists(DATABASE_FILE):
            try:
                with open(DATABASE_FILE, 'rb') as f:
                    db = pickle.load(f)
                self.known_encodings = list(db.values())
                self.known_ids = list(db.keys())
                print(f"[System] Loaded {len(self.known_ids)} identities.")
            except Exception as e:
                print(f"[Error] Database load failed: {e}")
        
        print("[System] Loading YOLOv8...")
        self.model = YOLO(YOLO_MODEL_PATH)
        self.track_history = {}

    def start_camera(self):
        """Initialize Video Capture"""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.update_frame()

    # BUSINESS LOGIC: SHIFT CALCULATION
    def get_current_shift_info(self):
        """Determine current shift and lateness status"""
        now = datetime.now()
        current_dt = datetime.strptime(now.strftime("%H:%M"), "%H:%M")

        for shift in SHIFTS:
            start_dt = datetime.strptime(shift["start"], "%H:%M")
            end_dt = datetime.strptime(shift["end"], "%H:%M")
            
            if start_dt <= current_dt <= end_dt:
                late_limit = start_dt + timedelta(minutes=LATE_THRESHOLD_MINUTES)
                status = "DUNG GIO" if current_dt <= late_limit else "DI MUON"
                return shift["name"], status
        
        return "NGOAI GIO", "---"

    #CORE LOOP: UI UPDAT
    def update_frame(self):
        """Main Loop: Capture -> Process -> Render"""
        if not self.is_running: return
        
        ret, frame = self.cap.read()
        if ret:
            # Process AI
            processed_frame = self.process_ai(frame)
            
            # Convert for Tkinter
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Resize to fit frame
            w = self.video_frame.winfo_width()
            h = self.video_frame.winfo_height()
            if w > 10 and h > 10:
                 ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(w, h))
                 self.video_label.configure(image=ctk_image)
                 self.video_label.image = ctk_image
                 
        # Update Shift Info
        shift_name, _ = self.get_current_shift_info()
        self.lbl_shift_info.configure(text=f"HIỆN TẠI: {shift_name}")
        
        self.after(10, self.update_frame)

    #PROCESSING ENGINE
    def process_ai(self, frame):
        """YOLO Tracking + Face Recognition + Logic"""
        # 1. Object Tracking
        results = self.model.track(frame, persist=True, conf=0.5, verbose=False, tracker="bytetrack.yaml")
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        else:
            boxes, track_ids = [], []

        current_time = time.time()
        
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            unique_id = "Unknown"
            
            # 2. Lazy Recognition Strategy
            if track_id in self.track_history and (current_time - self.track_history[track_id]['last_check'] < 2.0):
                unique_id = self.track_history[track_id]['unique_id']
            else:
                # Re-verify face
                rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_loc = [(y1, x2, y2, x1)]
                encs = face_recognition.face_encodings(rgb_small, face_loc, model='large')
                
                if encs and self.known_encodings:
                    dists = face_recognition.face_distance(self.known_encodings, encs[0])
                    best_idx = np.argmin(dists)
                    if dists[best_idx] < 0.45:
                        unique_id = self.known_ids[best_idx]
                
                self.track_history[track_id] = {'unique_id': unique_id, 'last_check': current_time}

            # 3. Visual & Logic Handling
            if unique_id != "Unknown":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Extract Data
                parts = unique_id.split('_')
                msv_text = parts[-1]
                name_text = " ".join(parts[:-1])

                shift_name, status_time = self.get_current_shift_info()
                checkin_key = f"{msv_text}_{shift_name}"
                
                if shift_name != "NGOAI GIO":
                    # Record Attendance (Once per shift)
                    if checkin_key not in self.checked_in_today:
                        self.record_attendance(name_text, msv_text, shift_name, status_time)
                        self.checked_in_today.add(checkin_key)
                        self.update_side_panel(name_text, msv_text, status_time, frame, box)
                    # Update UI (Debounced)
                    elif current_time - self.last_update_time > 2.0:
                         if name_text != self.last_update_name:
                             self.update_side_panel(name_text, msv_text, status_time, frame, box)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return frame

    #UI
    def update_side_panel(self, name, msv, status, frame, box):
        """Refresh Side Panel Info"""
        self.lbl_name.configure(text=name)
        self.lbl_msv.configure(text=f"MSV: {msv}")
        self.lbl_time.configure(text=datetime.now().strftime('%H:%M:%S'))

        color = "green" if status == "DUNG GIO" else ("red" if status == "DI MUON" else "gray")
        text_status = "ĐÚNG GIỜ" if status == "DUNG GIO" else ("ĐI MUỘN" if status == "DI MUON" else "NGOÀI GIỜ")
        
        self.lbl_status.configure(text=text_status, fg_color=color)

        # Crop & Show Avatar
        x1, y1, x2, y2 = box
        h, w, _ = frame.shape
        y1, y2 = max(0, y1 - 40), min(h, y2 + 40)
        x1, x2 = max(0, x1 - 30), min(w, x2 + 30)
        
        face_img = frame[y1:y2, x1:x2]
        if face_img.size > 0:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_ctk = ctk.CTkImage(light_image=face_pil, dark_image=face_pil, size=(200, 200))
            self.avatar_label.configure(image=face_ctk, text="")

        self.last_update_name = name
        self.last_update_time = time.time()

    #DATA PERSISTENCE
    def record_attendance(self, name, msv, shift, status):
        """Write to CSV"""
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        file_name = f"DiemDanh_{date_str}.csv"
        
        try:
            file_exists = os.path.isfile(file_name)
            with open(file_name, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['CaHoc', 'MSV', 'HoTen', 'GioVao', 'Ngay', 'TrangThai'])
                writer.writerow([shift, msv, name, now.strftime('%H:%M:%S'), date_str, status])
        except Exception as e:
            print(f"[Error] Writing CSV: {e}")

    # THREADED
    def toggle_menu(self, event=None):
        if self.menu_visible:
            self.btn_frame.pack_forget()
            self.menu_visible = False
        else:
            self.btn_frame.pack(side="bottom", pady=20, fill="x")
            self.menu_visible = True

    def action_add_face(self):
        # 1. Get Input
        dialog = AddStudentDialog(self)
        self.wait_window(dialog)
        
        if dialog.result:
            name, msv = dialog.result
            
            # 2. Stop Main Camera
            self.is_running = False 
            if self.cap: self.cap.release()
            
            self.video_label.configure(image=None, text="ĐANG XỬ LÝ... VUI LÒNG ĐỢI...")
            
            # 3. Execute in Thread
            threading.Thread(target=self._task_add_student, args=(name, msv), daemon=True).start()

    def _task_add_student(self, name, msv):
        time.sleep(1.0) # Safety Delay
        try:
            subprocess.run([sys.executable, "add_face.py", name, msv], check=True)
            
            self.after(0, lambda: self.video_label.configure(text="ĐANG CẬP NHẬT DATABASE..."))
            subprocess.run([sys.executable, "encode_database.py"], check=True)
            
            self.load_resources()
        except Exception as e:
            print(f"[Error] Task failed: {e}")

        self.after(0, self.restart_camera_safe)

    def action_reload_db(self):
        self.is_running = False
        if self.cap: self.cap.release()
        
        
        def _task():
            time.sleep(0.5)
            subprocess.run([sys.executable, "encode_database.py"])
            self.load_resources()
            self.after(0, self.restart_camera_safe)
            
        threading.Thread(target=_task, daemon=True).start()

    def restart_camera_safe(self):
        self.is_running = True
        self.start_camera()

    def action_open_excel(self):
        date_str = datetime.now().strftime('%Y-%m-%d')
        file_name = f"DiemDanh_{date_str}.csv"
        if os.path.exists(file_name):
            os.startfile(file_name)

    def close_app(self):
        self.is_running = False
        if self.cap: self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = KioskApp()
    app.mainloop()