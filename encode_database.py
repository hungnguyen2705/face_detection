import face_recognition
import os
import numpy as np
import pickle  


IMAGE_DIR = 'data_guong_mat'
DATABASE_FILE = 'face_database.pkl'

print("--- BẮT ĐẦU QUÁ TRÌNH ENCODING DATABASE ---")


temp_db = {}


print(f"Đang quét thư mục: {IMAGE_DIR}...")
for file in os.listdir(IMAGE_DIR):
    if file.endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(IMAGE_DIR, file)
        

        name = file.split('_')[0]
        
        try:
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img, model='large')
            
            if len(encs) > 0:
                if name not in temp_db:
                    temp_db[name] = []
                temp_db[name].append(encs[0])
                print(f" - Đã đọc: {file} (Gán cho: {name})")
                
        except Exception as e:
            print(f"Lỗi file {file}: {e}")

print("\nĐang tính toán vector trung bình cho từng người...")
final_database = {}

for name, vector_list in temp_db.items():
    if vector_list:
        avg_vector = np.mean(vector_list, axis=0)
        final_database[name] = avg_vector 
        print(f" + Đã tạo vector chuẩn cho: {name} (từ {len(vector_list)} ảnh)")


with open(DATABASE_FILE, 'wb') as f:
    pickle.dump(final_database, f)

print(f"\n--- HOÀN TẤT! Đã lưu database ra file: {DATABASE_FILE} ---")