import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time
from collections import deque, Counter

# =============================================================================
# 1. CẤU HÌNH HỆ THỐNG (USER CONFIG)
# =============================================================================

# Đường dẫn 
MODEL_PATH = "E:/Smart_Crib/yolov26_250e_1000pic_702010/train/weights/best.pt" # Đường dẫn file mô hình
#MODEL_PATH = "E:/Smart_Crib/yolov26_250e_1000pic_702010/train/weights/best_ncnn_model"
VIDEO_PATH = "E:/Smart_Crib/test/1100107673-preview.mp4" # Đường dẫn video test
#VIDEO_PATH = 0
# YOLO Class IDs 
CLS_FACE = 1
CLS_BODY = 0

# NGƯỠNG (THRESHOLDS) - QUAN TRỌNG
MAR_THRESHOLD_CRY = 0.27  # Miệng mở to hơn mức này -> KHÓC
EAR_THRESHOLD_SLEEP = 0.15  # Mắt nhắm nhỏ hơn mức này -> NGỦ (nếu không khóc)

# Số lượng frame để lọc nhiễu 
BUFFER_SIZE = 15

# Frame Skipping: FaceMesh chỉ chạy mỗi N frame (YOLO vẫn chạy liên tục)
# Tăng để FPS cao hơn, giảm để phản ứng nhanh hơn. Gợi ý: 2 hoặc 3
FACEMESH_SKIP = 2

# Cooldown trạng thái: số giây tối thiểu trước khi cho phép đổi sang trạng thái mới
# Tăng để ổn định hơn, giảm để phản ứng nhanh hơn. Gợi ý: 2.0 - 5.0
STATE_COOLDOWN = 3.0

# =============================================================================
# 2. KHỞI TẠO MODEL & THƯ VIỆN
# =============================================================================

print("[INFO] Đang tải YOLO model...")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"[ERROR] Không tìm thấy model tại {MODEL_PATH}. Lỗi: {e}")
    exit()

print("[INFO] Đang khởi tạo MediaPipe...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Bộ đệm lưu trạng thái
status_buffer = deque(maxlen=BUFFER_SIZE)


# =============================================================================
# 3. HÀM TÍNH TOÁN (EAR / MAR)
# =============================================================================

def calculate_ratio(landmarks, indices, w, h):
    """Hàm chung để tính EAR hoặc MAR dựa trên khoảng cách Euclidean"""
    # Lấy tọa độ các điểm
    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices])

    # Tính khoảng cách dọc (Vertical)
    # Với mắt: có 2 đường dọc. Với miệng: có 1 đường dọc chính
    if len(indices) == 6:  # Mắt (6 điểm)
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        hor = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * hor)

    elif len(indices) == 4:  # Miệng (4 điểm: Trái, Phải, Trên, Dưới)
        v = np.linalg.norm(pts[2] - pts[3])  # Trên - Dưới
        hor = np.linalg.norm(pts[0] - pts[1])  # Trái - Phải
        return v / hor
    return 0


def get_square_box(box, frame_w, frame_h, padding=0.2):
    """Mở rộng bounding box thành hình vuông"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w // 2, y1 + h // 2

    max_side = int(max(w, h) * (1 + padding))

    nx1 = max(0, cx - max_side // 2)
    ny1 = max(0, cy - max_side // 2)
    nx2 = min(frame_w, cx + max_side // 2)
    ny2 = min(frame_h, cy + max_side // 2)
    return int(nx1), int(ny1), int(nx2), int(ny2)


# =============================================================================
# 4. CHƯƠNG TRÌNH CHÍNH
# =============================================================================

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Định nghĩa Landmark MediaPipe
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    LIPS = [61, 291, 13, 14]  # 61: Left, 291: Right, 13: Upper, 14: Lower

    frame_count = 0
    mar_val = 0.0
    ear_val = 0.0
    black_display = np.zeros((500, 500, 3), dtype=np.uint8)

    # Cooldown: trạng thái được xác nhận cuối cùng hiển thị lên UI
    confirmed_status = "Waiting..."
    last_state_change_time = 0.0

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Hết video.")
            break

        frame_count += 1
        h, w, _ = frame.shape
        display = frame.copy()
        # --- BƯỚC 1: YOLO DETECT (FACE & BODY) ---
        results = model(frame, stream=True, verbose=False, conf=0.55, iou=0.5)

        faces = []
        bodies = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                coords = box.xyxy[0].cpu().numpy().astype(int)
                if cls_id == CLS_FACE:
                    faces.append(coords)
                    cv2.rectangle(display, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
                    cv2.putText(display, f"Face: {int(conf * 100)}%", (coords[0], max(10, coords[1] - 5)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif cls_id == CLS_BODY:
                    bodies.append(coords)
                    cv2.rectangle(display, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 0), 2)
                    cv2.putText(display, f"Body: {int(conf * 100)}%", (coords[0], max(10, coords[1] - 5)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # --- BƯỚC 2: LOGIC XỬ LÝ TRẠNG THÁI ---

        current_status = "Unknown"
        current_color = (200, 200, 200)
        # mar_val / ear_val được khai báo trước vòng lặp, giữ nguyên khi frame bị skip

        # Ưu tiên 1: Nằm úp (Thấy Body mà không thấy Face)
        if len(bodies) > 0 and len(faces) == 0:
            current_status = "NGUY HIEM: UP (Prone)"
            current_color = (0, 0, 255)  # Đỏ
            status_buffer.append(current_status)

        # Ưu tiên 2: Phân tích FaceMesh (Nếu thấy mặt)
        elif len(faces) > 0:
            # Lấy mặt lớn nhất
            face_box = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            nx1, ny1, nx2, ny2 = get_square_box(face_box, w, h)

            # --- FRAME SKIP B1: Chỉ chạy FaceMesh mỗi FACEMESH_SKIP frame ---
            if frame_count % FACEMESH_SKIP == 0:
                black_display[:] = 0  # Reset về đen chỉ khi sắp vẽ lại mesh
                face_roi = frame[ny1:ny2, nx1:nx2]
                if face_roi.size > 0:
                    roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    mesh_res = face_mesh.process(roi_rgb)

                    if mesh_res.multi_face_landmarks:
                        for fl in mesh_res.multi_face_landmarks:
                            # Vẽ lên cửa sổ đen
                            mp_drawing.draw_landmarks(
                                black_display, fl, mp_face_mesh.FACEMESH_TESSELATION,
                                None, mp_drawing_styles.get_default_face_mesh_tesselation_style())

                            # Tính toán EAR / MAR (cập nhật biến ngoài để giữ giá trị)
                            roi_h, roi_w, _ = face_roi.shape
                            landmarks = fl.landmark

                            ear_left = calculate_ratio(landmarks, LEFT_EYE, roi_w, roi_h)
                            ear_right = calculate_ratio(landmarks, RIGHT_EYE, roi_w, roi_h)
                            ear_val = (ear_left + ear_right) / 2.0
                            mar_val = calculate_ratio(landmarks, LIPS, roi_w, roi_h)

                            # --- LOGIC QUYẾT ĐỊNH (CORE LOGIC) ---
                            if mar_val > MAR_THRESHOLD_CRY:
                                status_temp = "KHOC (Crying)"
                            elif ear_val < EAR_THRESHOLD_SLEEP:
                                status_temp = "NGU (Sleeping)"
                            else:
                                status_temp = "THUC (Awake)"

                            status_buffer.append(status_temp)
            # Frame bị skip: ear_val/mar_val cũ vẫn còn, không cần append thêm vào buffer

        # Ưu tiên 3: Nôi trống (Không có người hoặc không thấy gì)
        else:
            status_buffer.append("NOI TRONG (Empty)")

        # --- BƯỚC 3: LỌC TRẠNG THÁI (VOTING) ---
        if len(status_buffer) > 0:
            final_status = Counter(status_buffer).most_common(1)[0][0]
        else:
            final_status = "Waiting..."

        # Chọn màu hiển thị theo final_status (raw voting)
        if "KHOC" in final_status or "UP" in final_status:
            current_color = (0, 0, 255)  # Đỏ
        elif "NGU" in final_status:
            current_color = (0, 255, 255)  # Vàng
        else:
            current_color = (0, 255, 0)  # Xanh

        # --- BƯỚC 3b: COOLDOWN - Chỉ đổi trạng thái sau STATE_COOLDOWN giây ---
        if final_status != confirmed_status:
            if time.time() - last_state_change_time >= STATE_COOLDOWN:
                confirmed_status = final_status
                last_state_change_time = time.time()

        # Cập nhật màu theo confirmed_status (ổn định)
        if "KHOC" in confirmed_status or "UP" in confirmed_status:
            current_color = (0, 0, 255)
        elif "NGU" in confirmed_status:
            current_color = (0, 255, 255)
        elif "TRONG" in confirmed_status:
            current_color = (150, 150, 150)  # Xám
        else:
            current_color = (0, 255, 0)

        # --- BƯỚC 4: TRỐNG (Đã xóa phát hiện người lớn để tối ưu FPS) ---

        # --- BƯỚC 5: HIỂN THỊ GIAO DIỆN (UI) ---

        # 1. Trạng thái chính (confirmed - ổn định)
        cv2.putText(display, f"STATUS: {confirmed_status}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, current_color, 3)
        # Raw voting (nhỏ, debug)
        cv2.putText(display, f"Raw: {final_status}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 2. Thanh Debug (MAR & EAR)
        # Vẽ background mờ cho thanh debug
        cv2.rectangle(display, (10, h - 80), (300, h - 10), (0, 0, 0), -1)

        # Thanh MAR (Độ mở miệng) - Màu Đỏ
        mar_width = int(min(mar_val, 1.0) * 200)
        cv2.rectangle(display, (20, h - 60), (20 + mar_width, h - 45), (0, 0, 255), -1)
        cv2.putText(display, f"Mouth: {mar_val:.2f} (Cry > {MAR_THRESHOLD_CRY})", (20, h - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Thanh EAR (Độ mở mắt) - Màu Vàng
        ear_width = int(min(ear_val, 1.0) * 200)
        cv2.rectangle(display, (20, h - 25), (20 + ear_width, h - 10), (0, 255, 255), -1)
        cv2.putText(display, f"Eye: {ear_val:.2f} (Sleep < {EAR_THRESHOLD_SLEEP})", (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 3. FPS
        process_time = time.time() - start_time
        fps = int(1.0 / process_time) if process_time > 0 else 0
        cv2.putText(display, f"FPS: {fps}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show windows
        cv2.imshow("Smart Crib Monitor", display)
        cv2.imshow("3D Face Mesh", black_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
