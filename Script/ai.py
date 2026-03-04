import cv2
from inference import get_model
import supervision as sv

# --- PHẦN ĐÃ CẬP NHẬT THÔNG SỐ TỪ ẢNH CỦA BẠN ---
# Model ID lấy từ: --project-id baby-sleep-detecting-dhxdw và --model-version 7
MODEL_ID = "baby-sleep-detecting-dhxdw/7"

# API Key lấy từ: --api-key aTLVETUHNTAfjfFHPTZE
API_KEY = "aTLVETUHNTAfjfFHPTZE"


# ------------------------------------------------

def run_detection():
    # 1. Tải mô hình về máy (lần đầu chạy sẽ cần internet để tải, các lần sau sẽ chạy offline)
    print("Đang tải model...")
    model = get_model(model_id=MODEL_ID, api_key=API_KEY)

    # 2. Mở camera (số 0 là webcam, hoặc thay bằng đường dẫn video nếu muốn test file)
    cap = cv2.VideoCapture('istockphoto-2194381495-640_adpp_is.mp4')

    # Cấu hình kích thước khung hình camera (tùy chọn, để chạy mượt hơn trên Pi)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Công cụ vẽ khung (bounding box) và nhãn (label)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    print("Đang chạy camera. Nhấn 'q' để thoát.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được camera.")
            break

        # 3. Gửi ảnh vào mô hình để nhận diện
        results = model.infer(frame)[0]

        # Chuyển đổi kết quả
        detections = sv.Detections.from_inference(results)

        # 4. Vẽ kết quả lên màn hình
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # Hiển thị
        cv2.imshow("Phat hien giac ngu em be", annotated_frame)

        # Nhấn phím 'q' để dừng chương trình
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection()