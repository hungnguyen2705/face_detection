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
        self.title("HỆ THỐNG NHẬN DIỆN KHUÔN MẶT")
        self.geometry("450x380") 
        self.resizable(False, False)

        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.title_label = ctk.CTkLabel(self.main_frame, 
                                        text="HỆ THỐNG NHẬN DIỆN", 
                                        font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(pady=(20, 20))


        self.btn_add = ctk.CTkButton(self.main_frame, 
                                     text="1. Thêm khuôn mặt & Tự động Encode", 
                                     command=self.prompt_and_run_add_face, 
                                     height=40,
                                     fg_color="#D2691E", hover_color="#CD5C5C") 
        self.btn_add.pack(pady=10, padx=20, fill="x")


        self.btn_encode = ctk.CTkButton(self.main_frame, 
                                        text="2. Cập nhật Database (Thủ công)", 
                                        command=self.run_encode_db_thread,
                                        height=40)
        self.btn_encode.pack(pady=10, padx=20, fill="x")


        self.btn_run = ctk.CTkButton(self.main_frame, 
                                     text="3. Bắt đầu nhận diện", 
                                     command=self.run_identify,
                                     height=40,
                                     fg_color="#006400", hover_color="#008000")
        self.btn_run.pack(pady=10, padx=20, fill="x")


        self.status_label = ctk.CTkLabel(self, text="Sẵn sàng", font=ctk.CTkFont(size=14))
        self.status_label.pack(pady=(0, 10))


    
    def prompt_and_run_add_face(self):
        """Hàm chính: Hiện popup hỏi tên, sau đó chạy luồng thêm mặt + encode."""
        

        dialog = ctk.CTkInputDialog(text="Nhập tên người cần thêm (không dấu):", 
                                    title="Thêm khuôn mặt")
        name = dialog.get_input()


        if name:

            self.disable_buttons()

            threading.Thread(target=self._task_add_and_encode, 
                             args=(name,), 
                             daemon=True).start()
        else:
            self.update_status("Đã hủy thao tác.")

    def _task_add_and_encode(self, name):
        """Hàm này chạy trong luồng riêng."""
        

        self.update_status(f"Đang chụp ảnh cho {name}...")
        add_process = subprocess.run([sys.executable, "add_face.py", name])


        if add_process.returncode == 0:

            self.update_status("Đang encode database...")
            encode_process = subprocess.run([sys.executable, "encode_database.py"])
            
            if encode_process.returncode == 0:
                self.update_status("Hoàn tất! Database đã được cập nhật.")
            else:
                self.update_status("Lỗi: Encode thất bại!")
        else:

            self.update_status("Đã hủy. Database chưa được cập nhật.")

        self.enable_buttons()

    def run_encode_db_thread(self):
        """Chạy encode thủ công trong 1 luồng riêng."""
        self.disable_buttons()
        threading.Thread(target=self._task_encode_only, daemon=True).start()

    def _task_encode_only(self):
        self.update_status("Đang encode (thủ công)...")
        subprocess.run([sys.executable, "encode_database.py"])
        self.update_status("Encode thủ công hoàn tất.")
        self.enable_buttons()

    def run_identify(self):
        if not os.path.exists('face_database.pkl'):
             self.update_status("Lỗi: Cần chạy 'Cập nhật Database' trước!")
        else:
            self.update_status("Đang khởi động nhận diện...")
            subprocess.Popen([sys.executable, "identify.py"])

    
    def update_status(self, text):
        """Cập nhật thanh trạng thái (an toàn từ mọi luồng)."""
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