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
    """Tính Mouth Aspect Ratio (Tỷ lệ mở miệng)"""
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
    cap = cv2.VideoCapture("test")

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

        # ==========================================
        # 1. MEDIAPIPE PROCESSING (Ưu tiên cao nhất)
        # ==========================================
        pose_results = pose_detector.process(rgb_frame)
        face_results = face_mesh.process(rgb_frame)

        # Kiểm tra có body không (MediaPipe Pose)
        is_body_detected = pose_results.pose_landmarks is not None

        # Kiểm tra Face từ FaceMesh
        is_face_mesh_detected = False
        face_landmarks_list = []
        if face_results.multi_face_landmarks:
            is_face_mesh_detected = True
            face_landmarks_list = face_results.multi_face_landmarks[0].landmark

        # Backup: Kiểm tra mũi từ Pose
        is_nose_visible_in_pose = False
        if is_body_detected:
            nose_visibility = pose_results.pose_landmarks.landmark[0].visibility
            if nose_visibility > 0.5:
                is_nose_visible_in_pose = True

        # Kết luận: Có mặt hay không?
        is_face_visible = is_face_mesh_detected or is_nose_visible_in_pose

        # ==========================================
        # 2. ROBOFLOW PROCESSING (Hỗ trợ & cung cấp label)
        # ==========================================
        roboflow_label = ""
        detections = None
        is_roboflow_success = False

        try:
            results = roboflow_model.infer(frame)[0]
            detections = sv.Detections.from_inference(results)

            # Vẽ Box
            display_frame = box_annotator.annotate(scene=display_frame, detections=detections)
            display_frame = label_annotator.annotate(scene=display_frame, detections=detections)

            # Lấy label nếu có
            if len(detections.class_id) > 0:
                roboflow_label = detections.data['class_name'][0]
                is_roboflow_success = True

        except Exception as e:
            print(f"⚠️ Lỗi Roboflow: {e}")

        # ==========================================
        # 3. LOGIC QUYẾT ĐỊNH (Ưu tiên MediaPipe, Roboflow hỗ trợ)
        # ==========================================
        status_text = ""
        status_color = (0, 255, 0)

        # TRƯỜNG HỢP 1: CÓ BODY NHƯNG KHÔNG CÓ MẶT → NẰM ÚP
        if is_body_detected and not is_face_visible:
            missing_face_counter += 1
            if missing_face_counter > ALARM_CONFIRM_FRAMES:
                status_text = "CANH BAO: NAM UP (NGUY HIEM)!"
                status_color = (0, 0, 255)

                # Vẽ skeleton đỏ
                mp_drawing.draw_landmarks(
                    display_frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            else:
                status_text = f"⏳ Kiem tra nam up... ({missing_face_counter}/{ALARM_CONFIRM_FRAMES})"
                status_color = (0, 255, 255)

        # TRƯỜNG HỢP 2: CÓ MẶT → KIỂM TRA KHÓC & TRẠNG THÁI
        elif is_face_visible:
            missing_face_counter = 0

            # A. Kiểm tra KHÓC
            is_crying = False
            mar_value = 0
            if is_face_mesh_detected:
                mar_value = calculate_mar(face_landmarks_list)
                if mar_value > MAR_THRESHOLD:
                    is_crying = True
                    status_text = f"BE DANG KHOC/NGAP (MAR: {mar_value:.2f})"
                    status_color = (0, 165, 255)

                    # Vẽ face mesh
                    mp_drawing.draw_landmarks(
                        image=display_frame,
                        landmark_list=face_results.multi_face_landmarks[0],
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

            # B. Nếu không khóc → Dùng label từ Roboflow (nếu có)
            if not is_crying:
                if is_roboflow_success and roboflow_label:
                    status_text = f"✅ Trang thai: {roboflow_label}"
                else:
                    status_text = "✅ Trang thai: An toan"
                status_color = (0, 255, 0)

        # TRƯỜNG HỢP 3: KHÔNG CÓ NGƯỜI (cả MediaPipe lẫn Roboflow đều không thấy)
        else:
            missing_face_counter = 0
            # Kiểm tra Roboflow có thấy không (fallback)
            if is_roboflow_success:
                status_text = f"✅ Roboflow: {roboflow_label} (MediaPipe miss)"
                status_color = (0, 255, 255)
            else:
                status_text = "Khong phat hien nguoi"
                status_color = (128, 128, 128)

        # ==========================================
        # UI - THANH THÔNG BÁO
        # ==========================================
        bar_height = 80
        cv2.rectangle(display_frame,
                      (0, frame_height - bar_height),
                      (frame_width, frame_height),
                      (0, 0, 0), -1)

        cv2.putText(display_frame, status_text,
                    (10, frame_height - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Debug info
        debug_mp = f"MediaPipe: Body={'Yes' if is_body_detected else 'No'} | Face={'Yes' if is_face_visible else 'No'}"
        debug_rf = f"Roboflow: {roboflow_label if is_roboflow_success else 'No detection'}"

        cv2.putText(display_frame, debug_mp,
                    (10, frame_height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(display_frame, debug_rf,
                    (10, frame_height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Smart Baby Monitor v3 - Hybrid", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_system()