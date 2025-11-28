import customtkinter as ctk
import subprocess
import sys
import os
import threading

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FACE RECOGNITION SYSTEM")
        self.geometry("450x350")
        self.resizable(False, False)

        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.title_label = ctk.CTkLabel(self.main_frame, 
                                        text="HỆ THỐNG NHẬN DIỆN", 
                                        font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(pady=(20, 20))

        # 
        self.btn_add = ctk.CTkButton(self.main_frame, 
                                     text="1. Thêm khuôn mặt mới (One-Shot)", 
                                     command=self.prompt_and_run_add_face,
                                     height=40,
                                     fg_color="#D2691E", hover_color="#CD5C5C")
        self.btn_add.pack(pady=10, padx=20, fill="x")

        # 
        self.btn_encode = ctk.CTkButton(self.main_frame, 
                                        text="2. Cập nhật Database (Thủ công)", 
                                        command=self.run_encode_db_thread,
                                        height=40)
        self.btn_encode.pack(pady=10, padx=20, fill="x")

        # 
        self.btn_run = ctk.CTkButton(self.main_frame, 
                                     text="3. Bắt đầu Nhận diện (Tracking)", 
                                     command=self.run_identify,
                                     height=40,
                                     fg_color="#006400", hover_color="#008000")
        self.btn_run.pack(pady=10, padx=20, fill="x")

        self.status_label = ctk.CTkLabel(self, text="Sẵn sàng")
        self.status_label.pack(pady=(0, 10))

    #
    def prompt_and_run_add_face(self):
        dialog = ctk.CTkInputDialog(text="Nhập tên người cần thêm:", title="Thêm khuôn mặt")
        name = dialog.get_input()
        if name:
            self.disable_buttons()

            threading.Thread(target=self._task_add_and_encode, args=(name,), daemon=True).start()
        else:
            self.update_status("Đã hủy.")

    def _task_add_and_encode(self, name):
        self.update_status(f"Đang chụp ảnh cho {name}...")
        # Chụp ảnh
        p1 = subprocess.run([sys.executable, "add_face.py", name])
        
        if p1.returncode == 0:
            self.update_status("Đang Encode lại dữ liệu...")
            # Encode
            p2 = subprocess.run([sys.executable, "encode_database.py"])
            if p2.returncode == 0:
                self.update_status("Thành công! Đã sẵn sàng.")
            else:
                self.update_status("Lỗi khi Encode!")
        else:
            self.update_status("Đã hủy thêm mặt.")
        
        self.enable_buttons()

    def run_encode_db_thread(self):
        self.disable_buttons()
        threading.Thread(target=self._task_encode_only, daemon=True).start()

    def _task_encode_only(self):
        self.update_status("Đang cập nhật...")
        subprocess.run([sys.executable, "encode_database.py"])
        self.update_status("Đã cập nhật xong.")
        self.enable_buttons()

    def run_identify(self):
        if not os.path.exists('face_database.pkl'):
             self.update_status("Lỗi: Chưa có Database!")
        else:
            self.update_status("Đang khởi động Tracking...")
            
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