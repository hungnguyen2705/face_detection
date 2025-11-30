# HỆ THỐNG ĐIỂM DANH SINH VIÊN THÔNG MINH (SMART ATTENDANCE SYSTEM)

## 1. Giới thiệu (Introduction)
Dự án xây dựng hệ thống điểm danh tự động sử dụng công nghệ Thị giác máy tính (Computer Vision) và Học sâu (Deep Learning). Hệ thống có khả năng nhận diện khuôn mặt theo thời gian thực, theo dõi đối tượng (Tracking), phân loại ca học và tự động ghi nhận trạng thái (Đúng giờ/Muộn) vào file Excel.

**Tính năng chính:**
* Nhận diện khuôn mặt Real-time với độ chính xác cao.
* Check-in tự động, phân chia 4 ca học trong ngày.
* Cảnh báo đi muộn ngay trên màn hình.
* Giao diện Kiosk (Full-screen) hiện đại, thân thiện.
* Quản lý danh sách sinh viên và xuất báo cáo Excel.

---

## 2. Công nghệ sử dụng (Tech Stack)

### Ngôn ngữ & Môi trường
* **Python 3.10+**: Ngôn ngữ lập trình chính.
* **Anaconda/Virtualenv**: Quản lý môi trường ảo.

### Thư viện lõi (Core Libraries)
* **Ultralytics YOLOv8**: Phát hiện (Detection) và Theo dõi (Tracking) khuôn mặt.
* **Dlib & Face_Recognition**: Trích xuất đặc trưng khuôn mặt (Feature Extraction).
* **OpenCV**: Xử lý ảnh và luồng video.
* **NumPy**: Xử lý toán học ma trận và vector.
* **CustomTkinter**: Xây dựng giao diện người dùng (GUI) hiện đại.
* **Pandas/CSV**: Xử lý và lưu trữ dữ liệu điểm danh.

---

## 3. Cài đặt và Chạy (Installation)

1.  **Clone dự án:**
    ```bash
    git clone [https://github.com/your-username/face-attendance-system.git](https://github.com/your-username/face-attendance-system.git)
    cd face-attendance-system
    ```

2.  **Cài đặt thư viện:**
    *Lưu ý: Cần cài đặt Visual Studio C++ Build Tools trước để biên dịch Dlib.*
    ```bash
    pip install -r requirements.txt
    ```

3.  **Chạy hệ thống:**
    ```bash
    python main.py
    ```

---

## 4. Cấu trúc Dự án (Project Structure)

* **`main.py`**: Chương trình chính (Main Entry Point). Chứa giao diện Kiosk, logic xử lý luồng video, logic điểm danh và quản lý đa luồng (Multithreading).
* **`add_face.py`**: Module thu thập dữ liệu. Chụp 10 ảnh mẫu của sinh viên từ Camera để làm cơ sở dữ liệu.
* **`encode_database.py`**: Module mã hóa dữ liệu. Đọc ảnh thô -> Trích xuất đặc trưng -> Lưu thành file nhị phân `face_database.pkl`.
* **`data_guong_mat/`**: Thư mục chứa ảnh gốc của sinh viên (Dataset).
* **`runs/`**: Thư mục chứa trọng số mô hình YOLOv8 (`best.pt`) đã được Fine-tune cho nhận diện khuôn mặt.

---

## 5. Các Thuật toán Cốt lõi & Code Minh họa (Core Algorithms)

Hệ thống hoạt động dựa trên sự kết hợp của 5 thuật toán chính sau đây:

### 5.1. Thuật toán Phát hiện & Theo dõi (Detection & Tracking)
* **Mô hình:** YOLOv8 (You Only Look Once) kết hợp ByteTrack.
* **Cơ chế:**
    * **YOLOv8 (CSPDarknet):** Sử dụng mạng nơ-ron tích chập (CNN) để dự đoán tọa độ khung bao (Bounding Box) của khuôn mặt trong ảnh.
    * **ByteTrack:** Sử dụng thuật toán Kalman Filter để dự đoán vị trí tiếp theo và thuật toán Hungarian để ghép nối ID đối tượng giữa các khung hình, giúp hệ thống "nhớ" được người đó là ai mà không cần nhận diện lại liên tục.
* **Code thực tế (trong `main.py`):**
    ```python
    # persist=True: Kích hoạt chế độ theo dõi, giữ ID qua các frame
    # tracker="bytetrack.yaml": Sử dụng thuật toán ByteTrack
    results = self.model.track(frame, persist=True, conf=0.5, verbose=False, tracker="bytetrack.yaml")
    
    # Lấy tọa độ hộp bao (x1, y1, x2, y2) và ID theo dõi
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
    ```

### 5.2. Thuật toán Căn chỉnh khuôn mặt (Face Alignment)
* **Thư viện:** Dlib.
* **Cơ chế:** Xác định 68 điểm mốc (Landmarks) trên khuôn mặt (mắt, mũi, miệng, viền hàm). Sau đó thực hiện phép biến đổi hình học (Affine Transformation) để xoay thẳng khuôn mặt về vị trí chuẩn trước khi trích xuất đặc trưng.
* **Code thực tế (Được gọi ngầm trong thư viện `face_recognition`):**
    ```python
    # Hàm này tự động tìm landmarks và căn chỉnh ảnh (Image Alignment)
    # trước khi trả về vector đặc trưng
    encs = face_recognition.face_encodings(rgb_small, face_loc, model='large')
    ```

### 5.3. Thuật toán Trích xuất Đặc trưng (Feature Extraction)
* **Mô hình:** ResNet-34 (Deep Residual Network).
* **Cơ chế:** Biến đổi một hình ảnh khuôn mặt ($128 \times 128$ pixels) thành một Vector đặc trưng 128 chiều ($128$-d embedding). Vector này đại diện cho các đặc điểm duy nhất của khuôn mặt, bất biến với ánh sáng.
* **Code thực tế (trong `add_face.py` & `main.py`):**
    ```python
    # Input: Ảnh RGB chứa khuôn mặt
    # Output: List chứa 128 số thực (Vector)
    # model='large': Sử dụng mạng ResNet đầy đủ để đạt độ chính xác cao nhất
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model='large')
    ```

### 5.4. Thuật toán Tối ưu hóa Dữ liệu (Vector Averaging)
* **Mục đích:** Giảm nhiễu (Noise Reduction) và tăng độ ổn định.
* **Cơ chế:** Thay vì lưu 10 vector của 10 bức ảnh chụp khác nhau, hệ thống tính trung bình cộng (Mean) của các vector này để tạo ra một "Super Vector" đại diện duy nhất cho sinh viên đó.
* **Code thực tế (trong `encode_database.py`):**
    ```python
    # temp_db[unique_id] là danh sách chứa 10 vector của 1 sinh viên
    for uid, vector_list in temp_db.items():
        if vector_list:
            # Tính trung bình cộng dọc theo trục 0
            avg_vector = np.mean(vector_list, axis=0)
            final_database[uid] = avg_vector
    ```

### 5.5. Thuật toán So khớp & Định danh (Matching)
* **Phương pháp:** Khoảng cách Euclidean (L2 Norm).
* **Cơ chế:** Tính khoảng cách hình học giữa Vector khuôn mặt hiện tại và các Vector trong Database. Khoảng cách càng nhỏ nghĩa là độ tương đồng càng cao.
* **Ngưỡng (Threshold):** `0.45` (Nếu khoảng cách < 0.45 thì coi là cùng một người).
* **Code thực tế (trong `main.py`):**
    ```python
    # Tính khoảng cách Euclidean với toàn bộ database
    dists = face_recognition.face_distance(self.known_encodings, encs[0])
    
    # Tìm khoảng cách nhỏ nhất
    best_idx = np.argmin(dists)
    
    # Kiểm tra ngưỡng chấp nhận
    if dists[best_idx] < 0.45:
        name = self.known_names[best_idx] # Định danh thành công
    ```

---

## 6. Logic Nghiệp vụ (Business Logic)

Hệ thống xử lý điểm danh dựa trên thời gian thực:
1.  **Chia Ca:** 4 Ca học được định nghĩa trong mảng `SHIFTS` với `start` và `end`.
2.  **Độ trễ (Late Threshold):** Cho phép muộn `15 phút` so với giờ bắt đầu (`start`).
3.  **Trạng thái:**
    * `Thời gian hiện tại <= Giờ bắt đầu + 15p` -> **DUNG GIO**.
    * `Thời gian hiện tại > Giờ bắt đầu + 15p` -> **DI MUON**.
    * Ngoài giờ học -> **NGOAI GIO**.

---

