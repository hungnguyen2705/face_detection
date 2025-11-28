import customtkinter as ctk
import subprocess
import sys
import os
import threading
from datetime import datetime

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("HỆ THỐNG ĐIỂM DANH AI")
        self.geometry("450x450") 
        self.resizable(False, False)

        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.title_label = ctk.CTkLabel(self.main_frame, 
                                        text="QUẢN LÝ ĐIỂM DANH", 
                                        font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(pady=(20, 20))


        self.btn_add = ctk.CTkButton(self.main_frame, 
                                     text="1. Thêm nhân viên mới", 
                                     command=self.prompt_and_run_add_face,
                                     height=40, fg_color="#D2691E", hover_color="#CD5C5C")
        self.btn_add.pack(pady=10, padx=20, fill="x")


        self.btn_encode = ctk.CTkButton(self.main_frame, 
                                        text="2. Cập nhật dữ liệu khuôn mặt", 
                                        command=self.run_encode_db_thread,
                                        height=40)
        self.btn_encode.pack(pady=10, padx=20, fill="x")


        self.btn_run = ctk.CTkButton(self.main_frame, 
                                     text="3. BẮT ĐẦU CHẤM CÔNG", 
                                     command=self.run_identify,
                                     height=50, # Nút to hơn
                                     font=ctk.CTkFont(size=16, weight="bold"),
                                     fg_color="#006400", hover_color="#008000")
        self.btn_run.pack(pady=10, padx=20, fill="x")
        

        self.btn_open_excel = ctk.CTkButton(self.main_frame, 
                                     text="4. Mở file điểm danh hôm nay", 
                                     command=self.open_attendance_file,
                                     height=30,
                                     fg_color="#4682B4", hover_color="#5F9EA0")
        self.btn_open_excel.pack(pady=10, padx=20, fill="x")

        self.status_label = ctk.CTkLabel(self, text="Sẵn sàng")
        self.status_label.pack(pady=(0, 10))


    def open_attendance_file(self):
        
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        file_name = f"DiemDanh_{date_str}.csv"
        
        if os.path.exists(file_name):
            self.update_status(f"Đang mở: {file_name}")
            os.startfile(file_name) 
        else:
            self.update_status("Chưa có dữ liệu điểm danh hôm nay!")

    def prompt_and_run_add_face(self):
        dialog = ctk.CTkInputDialog(text="Nhập tên nhân viên:", title="Thêm mới")
        name = dialog.get_input()
        if name:
            self.disable_buttons()
            threading.Thread(target=self._task_add_and_encode, args=(name,), daemon=True).start()
        else:
            self.update_status("Đã hủy.")

    def _task_add_and_encode(self, name):
        self.update_status(f"Đang chụp ảnh: {name}...")
        p1 = subprocess.run([sys.executable, "add_face.py", name])
        if p1.returncode == 0:
            self.update_status("Đang mã hóa dữ liệu...")
            p2 = subprocess.run([sys.executable, "encode_database.py"])
            if p2.returncode == 0:
                self.update_status("Thêm nhân viên thành công!")
            else:
                self.update_status("Lỗi cập nhật data!")
        else:
            self.update_status("Đã hủy.")
        self.enable_buttons()

    def run_encode_db_thread(self):
        self.disable_buttons()
        threading.Thread(target=self._task_encode_only, daemon=True).start()

    def _task_encode_only(self):
        self.update_status("Đang cập nhật...")
        subprocess.run([sys.executable, "encode_database.py"])
        self.update_status("Cập nhật xong.")
        self.enable_buttons()

    def run_identify(self):
        if not os.path.exists('face_database.pkl'):
             self.update_status("Lỗi: Chưa có dữ liệu nhân viên!")
        else:
            self.update_status("Đang chạy máy chấm công...")
            subprocess.Popen([sys.executable, "identify.py"])

    def update_status(self, text):
        self.status_label.configure(text=text)

    def disable_buttons(self):
        self.btn_add.configure(state="disabled")
        self.btn_encode.configure(state="disabled")
        self.btn_run.configure(state="disabled")

    def enable_buttons(self):
        self.btn_add.configure(state="normal")
        self.btn_encode.configure(state="normal")
        self.btn_run.configure(state="normal")

if __name__ == "__main__":
    app = App()
    app.mainloop()