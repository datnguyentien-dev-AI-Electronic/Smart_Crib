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
ALARM_CONFIRM_FRAMES = 80  # Số frame cần confirm (30 frames ≈ 1 giây với 30 FPS)
ALARM_CONFIRM_TIME = 10  # Thời gian cần confirm (giây) - Ưu tiên hơn frames
POSE_CONFIDENCE = 0.5
FACE_CONFIDENCE = 0.5
MAR_THRESHOLD = 0.3

# MediaPipe Setup
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Khởi tạo Models
pose_detector = mp_pose.Pose(min_detection_confidence=POSE_CONFIDENCE, min_tracking_confidence=POSE_CONFIDENCE)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=FACE_CONFIDENCE)

# Chỉ số Landmark cho Môi
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
    """Tính Mouth Aspect Ratio"""
    top = landmarks[UPPER_LIP]
    bottom = landmarks[LOWER_LIP]
    left = landmarks[LEFT_LIP]
    right = landmarks[RIGHT_LIP]

    v_dist = euclidean_distance(top, bottom)
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
    cap = cv2.VideoCapture("demo_lie_down.mp4")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Setup Supervision
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # BIẾN ĐẾM VÀ THỜI GIAN
    missing_face_counter = 0
    face_down_start_time = None  # Thời điểm bắt đầu phát hiện nằm úp
    face_down_duration = 0.0  # Tổng thời gian nằm úp (giây)

    print("🚀 Hệ thống đang chạy...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        frame_height, frame_width, _ = display_frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ==========================================
        # 1. ROBOFLOW - Phát hiện có người không
        # ==========================================
        roboflow_label = ""
        is_person_detected = False

        try:
            results = roboflow_model.infer(frame)[0]
            detections = sv.Detections.from_inference(results)

            display_frame = box_annotator.annotate(scene=display_frame, detections=detections)
            display_frame = label_annotator.annotate(scene=display_frame, detections=detections)

            if len(detections.class_id) > 0:
                is_person_detected = True
                roboflow_label = detections.data['class_name'][0]

        except Exception as e:
            print(f"⚠️ Roboflow error: {e}")

        # ==========================================
        # 2. MEDIAPIPE - Kiểm tra Body (Pose) + Face
        # ==========================================
        pose_results = pose_detector.process(rgb_frame)
        face_results = face_mesh.process(rgb_frame)

        # Kiểm tra có body (em bé) không
        has_body = pose_results.pose_landmarks is not None

        # Kiểm tra có landmarks mặt không
        has_face_landmarks = False
        face_landmarks_list = []

        if face_results.multi_face_landmarks:
            has_face_landmarks = True
            face_landmarks_list = face_results.multi_face_landmarks[0].landmark

        # ==========================================
        # 3. LOGIC QUYẾT ĐỊNH - KẾT HỢP 3 NGUỒN
        # ==========================================
        status_text = ""
        status_color = (0, 255, 0)

        # XÁC ĐỊNH CÓ EM BÉ KHÔNG (ưu tiên: Roboflow hoặc MediaPipe Pose)
        is_baby_present = is_person_detected or has_body

        # TRƯỜNG HỢP 1: KHÔNG CÓ EM BÉ
        if not is_baby_present:
            missing_face_counter = 0
            face_down_start_time = None
            face_down_duration = 0.0
            status_text = "❌ Khong phat hien em be"
            status_color = (128, 128, 128)

        # TRƯỜNG HỢP 2: CÓ EM BÉ + KHÔNG CÓ MẶT → NẰM ÚP
        elif is_baby_present and not has_face_landmarks:
            # Bắt đầu đếm thời gian nằm úp
            if face_down_start_time is None:
                face_down_start_time = time.time()

            face_down_duration = time.time() - face_down_start_time
            missing_face_counter += 1

            # Kiểm tra cả THỜI GIAN và SỐ FRAME
            if face_down_duration >= ALARM_CONFIRM_TIME or missing_face_counter > ALARM_CONFIRM_FRAMES:
                status_text = f"🚨 CANH BAO: NAM UP {face_down_duration:.1f}s!"
                status_color = (0, 0, 255)

                # Vẽ skeleton nếu có để cảnh báo
                if has_body:
                    mp_drawing.draw_landmarks(
                        display_frame,
                        pose_results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
            else:
                status_text = f"⏳ Kiem tra mat... {face_down_duration:.1f}s/{ALARM_CONFIRM_TIME}s ({missing_face_counter}/{ALARM_CONFIRM_FRAMES})"
                status_color = (0, 255, 255)

        # TRƯỜNG HỢP 3: CÓ EM BÉ + CÓ MẶT → An toàn, kiểm tra khóc & trạng thái
        elif is_baby_present and has_face_landmarks:
            # Reset bộ đếm khi thấy mặt
            missing_face_counter = 0
            face_down_start_time = None
            face_down_duration = 0.0

            # Kiểm tra KHÓC
            is_crying = False
            mar_value = calculate_mar(face_landmarks_list)

            if mar_value > MAR_THRESHOLD:
                is_crying = True
                status_text = f"😭 BE DANG KHOC/NGAP (MAR: {mar_value:.2f})"
                status_color = (0, 165, 255)

                # Vẽ face mesh
                mp_drawing.draw_landmarks(
                    image=display_frame,
                    landmark_list=face_results.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

            # Không khóc → Hiển thị trạng thái từ Roboflow
            if not is_crying:
                if roboflow_label:
                    status_text = f"✅ Trang thai: {roboflow_label}"
                else:
                    status_text = "✅ Trang thai: An toan"
                status_color = (0, 255, 0)

        # ==========================================
        # UI - THANH THÔNG BÁO
        # ==========================================
        bar_height = 80
        cv2.rectangle(display_frame,
                      (0, frame_height - bar_height),
                      (frame_width, frame_height),
                      (0, 0, 0), -1)

        # Status chính
        cv2.putText(display_frame, status_text,
                    (10, frame_height - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Debug info
        debug_detect = f"Roboflow: {roboflow_label if is_person_detected else 'No'} | Pose: {'Yes' if has_body else 'No'} | Baby: {'YES' if is_baby_present else 'NO'}"
        debug_face = f"Face Landmarks: {'DETECTED' if has_face_landmarks else 'NOT DETECTED'} | Face Down Time: {face_down_duration:.2f}s"

        cv2.putText(display_frame, debug_detect,
                    (10, frame_height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(display_frame, debug_face,
                    (10, frame_height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Smart Baby Monitor - Hybrid Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_system()