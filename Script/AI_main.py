import cv2
import mediapipe as mp
import math
from inference import get_model
import supervision as sv
import time

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
MODEL_ID = "baby-sleep-detecting-dhxdw/7"
API_KEY = "aTLVETUHNTAfjfFHPTZE"

# CẤU HÌNH CHỐNG NHIỄU & NGƯỠNG
ALARM_CONFIRM_FRAMES = 30
POSE_CONFIDENCE = 0.5
FACE_CONFIDENCE = 0.5
MAR_THRESHOLD = 0.3  # Ngưỡng mở miệng (Cần tinh chỉnh: > 0.4 - 0.5 thường là khóc/ngáp to)

# MediaPipe Setup
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh  # Đổi sang Face Mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Khởi tạo Models
pose_detector = mp_pose.Pose(min_detection_confidence=POSE_CONFIDENCE, min_tracking_confidence=POSE_CONFIDENCE)
# Refine_landmarks=True để lấy điểm mắt/môi chuẩn hơn
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=FACE_CONFIDENCE)

# Chỉ số Landmark cho Môi (MediaPipe chuẩn)
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_LIP = 61
RIGHT_LIP = 291


# ==========================================
# HÀM PHỤ TRỢ
# ==========================================
def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def calculate_mar(landmarks):
    """Tính Mouth Aspect Ratio (Tỷ lệ mở miệng)"""
    top = landmarks[UPPER_LIP]
    bottom = landmarks[LOWER_LIP]
    left = landmarks[LEFT_LIP]
    right = landmarks[RIGHT_LIP]

    # Khoảng cách dọc (mở miệng)
    v_dist = euclidean_distance(top, bottom)
    # Khoảng cách ngang (độ rộng miệng)
    h_dist = euclidean_distance(left, right)

    if h_dist == 0: return 0
    return v_dist / h_dist


def run_system():
    # Load Model Roboflow
    print("⏳ Đang tải model Roboflow...")
    try:
        roboflow_model = get_model(model_id=MODEL_ID, api_key=API_KEY)
        print("✅ Đã tải model thành công!")
    except Exception as e:
        print(f"❌ Lỗi tải model: {e}")
        return

    # Camera setup
    # cap = cv2.VideoCapture('1100107673-preview.mp4')
    cap = cv2.VideoCapture("Video Project 11.mp4")  # Dùng webcam hoặc đổi lại file video

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Setup Supervision
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # BIẾN ĐẾM
    missing_face_counter = 0

    print("🚀 Hệ thống đang chạy...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        frame_height, frame_width, _ = display_frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. MediaPipe Xử lý (Pose + FaceMesh)
        pose_results = pose_detector.process(rgb_frame)
        face_results = face_mesh.process(rgb_frame)

        # Kiểm tra cơ bản
        is_body_detected = pose_results.pose_landmarks is not None

        # Kiểm tra Face từ FaceMesh
        is_face_mesh_detected = False
        face_landmarks_list = []
        if face_results.multi_face_landmarks:
            is_face_mesh_detected = True
            face_landmarks_list = face_results.multi_face_landmarks[0].landmark

        # Logic Backup: Kiểm tra mũi từ Pose (phòng khi FaceMesh trượt nhưng Pose vẫn thấy mũi)
        is_nose_visible_in_pose = False
        if is_body_detected:
            nose_visibility = pose_results.pose_landmarks.landmark[0].visibility
            if nose_visibility > 0.5:
                is_nose_visible_in_pose = True

        # Kết luận: Có mặt hay không?
        is_face_really_visible = is_face_mesh_detected or is_nose_visible_in_pose

        # --- LOGIC QUYẾT ĐỊNH TRẠNG THÁI ---
        status_text = ""
        status_color = (0, 255, 0)  # Xanh (Mặc định An toàn)

        # TRƯỜNG HỢP 1: NẰM ÚP (Ưu tiên cao nhất)
        if is_body_detected and not is_face_really_visible:
            missing_face_counter += 1
            if missing_face_counter > ALARM_CONFIRM_FRAMES:
                status_text = "CANH BAO: NAM UP (NGUY HIEM)!"
                status_color = (0, 0, 255)  # Đỏ

                # Vẽ Skeleton cảnh báo
                mp_drawing.draw_landmarks(display_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255)))
            else:
                status_text = f"Check mat... ({missing_face_counter}/{ALARM_CONFIRM_FRAMES})"
                status_color = (0, 255, 255)  # Vàng

        # TRƯỜNG HỢP 2: AN TOÀN (Thấy mặt) -> Kiểm tra KHÓC & Inference
        elif is_face_really_visible:
            missing_face_counter = 0

            # A. Kiểm tra KHÓC (Dựa trên Face Mesh MAR)
            is_crying = False
            mar_value = 0
            if is_face_mesh_detected:
                mar_value = calculate_mar(face_landmarks_list)
                if mar_value > MAR_THRESHOLD:
                    is_crying = True
                    status_text = f" BE DANG KHOC/NGAP (MAR: {mar_value:.2f})"
                    status_color = (0, 165, 255)  # Cam

                    # Vẽ lưới mặt để debug (Tùy chọn)
                    mp_drawing.draw_landmarks(
                        image=display_frame,
                        landmark_list=face_results.multi_face_landmarks[0],
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            # B. Chạy Model Roboflow (Sleep/Awake Detection)
            # Vẫn chạy cái này để vẽ bounding box ngay cả khi khóc
            try:
                results = roboflow_model.infer(frame)[0]
                detections = sv.Detections.from_inference(results)

                # Vẽ Box
                display_frame = box_annotator.annotate(scene=display_frame, detections=detections)
                display_frame = label_annotator.annotate(scene=display_frame, detections=detections)

                # Nếu không khóc thì lấy label từ model (Sleep/Awake)
                if not is_crying:
                    # Lấy class name đầu tiên detected (nếu có)
                    if len(detections.class_id) > 0:
                        # Giả sử class_name trả về từ model
                        status_text = f"Trang thai: {detections.data['class_name'][0]}"
                        #status_text = f"Trang thai: An toan (Roboflow Status:)"
                    else:
                        status_text = "Trang thai: An toan"

            except Exception:
                if not is_crying:
                    status_text = "Loi AI Inference"

        # TRƯỜNG HỢP 3: KHÔNG CÓ NGƯỜI
        else:
            missing_face_counter = 0
            status_text = "Khong co nguoi"
            status_color = (128, 128, 128)  # Xám

        # ==========================================
        # VẼ THANH THÔNG BÁO (UI)
        # ==========================================
        bar_height = 50
        cv2.rectangle(display_frame,
                      (0, frame_height - bar_height),
                      (frame_width, frame_height),
                      (0, 0, 0), -1)

        cv2.putText(display_frame, status_text,
                    (10, frame_height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.imshow("Smart Baby Monitor", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_system()