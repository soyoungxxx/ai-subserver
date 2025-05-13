from ultralytics import YOLO
import cv2

class YOLOModel:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def predict(self, input_path, output_path):
        # 입력 파일 확장자 확인
        is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov'))

        if is_video:
            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("프레임 읽기 실패, grab 시도")
                    cap.grab()  # 다음 프레임으로 강제로 건너뜀
                    continue

                # 추론
                results = self.model(frame)

                # 결과 시각화
                annotated_frame = results[0].plot()

                out.write(annotated_frame)

            cap.release()
            out.release()

        else:
            # 이미지 추론
            img = cv2.imread(input_path)
            results = self.model(img)
            annotated = results[0].plot()
            cv2.imwrite(output_path, annotated)
