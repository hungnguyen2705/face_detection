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
from datetime import datetime
from PIL import Image, ImageTk
from ultralytics import YOLO


WORK_START_TIME = "08:00:00"
SAVE_DIR = 'data_guong_mat'
DATABASE_FILE = 'face_database.pkl'

def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_base_path()
YOLO_MODEL_PATH = os.path.join(BASE_PATH, r'runs/detect/train4/weights/best.pt')

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class KioskApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("HỆ THỐNG CHẤM CÔNG THÔNG MINH")
        
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height}")
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)


        self.video_frame = ctk.CTkFrame(self, fg_color="black")
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(fill="both", expand=True)
        
        self.video_label.bind("<Button-1>", self.toggle_menu)


        self.info_frame = ctk.CTkFrame(self, fg_color="#2b2b2b", corner_radius=0)
        self.info_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        
        self.info_frame.bind("<Button-1>", self.toggle_menu)


        ctk.CTkLabel(self.info_frame, text="THÔNG TIN CHECK-IN", font=("Arial", 24, "bold")).pack(pady=(40, 20))

        self.avatar_label = ctk.CTkLabel(self.info_frame, text="[Ảnh]", width=200, height=200, fg_color="#444")
        self.avatar_label.pack(pady=10)

        self.lbl_name = ctk.CTkLabel(self.info_frame, text="---", font=("Arial", 30, "bold"), text_color="white")
        self.lbl_name.pack(pady=(20, 5))

        self.lbl_time = ctk.CTkLabel(self.info_frame, text="--:--:--", font=("Arial", 60, "bold"), text_color="#00FFFF")
        self.lbl_time.pack(pady=5)

        self.lbl_status = ctk.CTkLabel(self.info_frame, text="SẴN SÀNG", font=("Arial", 20, "bold"), 
                                       fg_color="#444", corner_radius=10, width=200, height=50)
        self.lbl_status.pack(pady=30)
        
        self.btn_frame = ctk.CTkFrame(self.info_frame, fg_color="transparent")
        
        ctk.CTkButton(self.btn_frame, text="Thêm NV Mới", command=self.action_add_face, height=40).pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.btn_frame, text="Cập nhật Data", command=self.action_reload_db, height=40).pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.btn_frame, text="Mở Excel", command=self.action_open_excel, fg_color="green", height=40).pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(self.btn_frame, text="Thoát", command=self.close_app, fg_color="red", height=40).pack(fill="x", padx=20, pady=5)
        
        self.menu_visible = False 


        self.is_running = True
        self.cap = None
        self.checked_in_today = set()
        self.last_update_name = "" 
        self.last_update_time = 0

        self.load_resources()
        self.start_camera()

    def load_resources(self):
        print("--- Đang khởi động hệ thống ---")
        self.known_encodings = []
        self.known_names = []
        if os.path.exists(DATABASE_FILE):
            try:
                with open(DATABASE_FILE, 'rb') as f:
                    db = pickle.load(f)
                self.known_encodings = list(db.values())
                self.known_names = list(db.keys())
                print(f"Đã load {len(self.known_names)} hồ sơ.")
            except:
                print("Database lỗi.")
        
        print("Loading YOLOv8...")
        self.model = YOLO(YOLO_MODEL_PATH)
        self.track_history = {}

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def toggle_menu(self, event=None):
        """Hàm Bật/Tắt các nút chức năng"""
        if self.menu_visible:
            self.btn_frame.pack_forget()
            self.menu_visible = False
        else:
            self.btn_frame.pack(side="bottom", pady=20, fill="x")
            self.menu_visible = True

    def update_frame(self):
        if not self.is_running: return
        
        ret, frame = self.cap.read()
        if ret:
            processed_frame = self.process_ai(frame)
            
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            
            video_w = self.video_frame.winfo_width()
            video_h = self.video_frame.winfo_height()
            
            if video_w > 10 and video_h > 10:
                 ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(video_w, video_h))
                 self.video_label.configure(image=ctk_image)
                 self.video_label.image = ctk_image
        
        self.after(10, self.update_frame)

    def process_ai(self, frame):
        
        results = self.model.track(frame, persist=True, conf=0.5, verbose=False, tracker="bytetrack.yaml")
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        else:
            boxes = []
            track_ids = []

        current_time = time.time()
        
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            name = "Unknown"
            
            
            if track_id in self.track_history and (current_time - self.track_history[track_id]['last_check'] < 2.0):
                name = self.track_history[track_id]['name']
            else:
                rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_loc = [(y1, x2, y2, x1)]
                encs = face_recognition.face_encodings(rgb_small, face_loc, model='large')
                
                if encs:
                    if len(self.known_encodings) > 0:
                        dists = face_recognition.face_distance(self.known_encodings, encs[0])
                        best_idx = np.argmin(dists)
                        if dists[best_idx] < 0.45:
                            name = self.known_names[best_idx]
                
                self.track_history[track_id] = {'name': name, 'last_check': current_time}

            if name != "Unknown":
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                
                if name not in self.checked_in_today:
                    self.record_attendance(name)
                    self.checked_in_today.add(name)
                    self.update_side_panel(name, frame, box)
                elif current_time - self.last_update_time > 2.0:
                     if name != self.last_update_name:
                         self.update_side_panel(name, frame, box)
            else:
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return frame

    def update_side_panel(self, name, frame, box):
        
        self.lbl_name.configure(text=name)
        
        now = datetime.now()
        time_str = now.strftime('%H:%M:%S')
        self.lbl_time.configure(text=time_str)

        
        limit_time = datetime.strptime(WORK_START_TIME, "%H:%M:%S").time()
        current_t = now.time()
        
        if current_t > limit_time:
            self.lbl_status.configure(text="ĐI MUỘN", fg_color="red")
        else:
            self.lbl_status.configure(text="ĐÚNG GIỜ", fg_color="green")

        x1, y1, x2, y2 = box
        h, w, _ = frame.shape
        y1 = max(0, y1 - 40); y2 = min(h, y2 + 40)
        x1 = max(0, x1 - 30); x2 = min(w, x2 + 30)
        
        face_img = frame[y1:y2, x1:x2]
        if face_img.size > 0:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_ctk = ctk.CTkImage(light_image=face_pil, dark_image=face_pil, size=(200, 200))
            self.avatar_label.configure(image=face_ctk, text="")

        self.last_update_name = name
        self.last_update_time = time.time()

    def record_attendance(self, name):
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        
        limit_time = datetime.strptime(WORK_START_TIME, "%H:%M:%S").time()
        current_time = now.time()
        status = "Dung Gio"
        if current_time > limit_time:
            status = "DI MUON"

        file_name = f"DiemDanh_{date_str}.csv"
        file_exists = os.path.isfile(file_name)
        try:
            with open(file_name, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['HoTen', 'GioVao', 'Ngay', 'TrangThai'])
                writer.writerow([name, time_str, date_str, status])
        except:
            pass

    def action_add_face(self):
        self.cap.release()
        dialog = ctk.CTkInputDialog(text="Nhập tên nhân viên:", title="Thêm mới")
        name = dialog.get_input()
        if name:
            subprocess.run([sys.executable, "add_face.py", name])
            subprocess.run([sys.executable, "encode_database.py"])
            self.load_resources()
        self.start_camera()

    def action_reload_db(self):
        self.cap.release()
        subprocess.run([sys.executable, "encode_database.py"])
        self.load_resources()
        self.start_camera()

    def action_open_excel(self):
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        file_name = f"DiemDanh_{date_str}.csv"
        if os.path.exists(file_name):
            os.startfile(file_name)

    def close_app(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = KioskApp()
    app.mainloop()