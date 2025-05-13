from ultralytics import YOLO
import cv2
import numpy as np

class YOLOModel:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def predict(self, input_path, output_path, polygon_data):
        # 입력 파일 확장자 확인
        is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov'))

        # polygon 좌표 numpy 배열로 변환
        pts = np.array(
            [[int(p.x), int(p.y)] for p in polygon_data.polygon],
            dtype=np.int32
        )
        print(pts)

        # 영상일 경우
        if is_video:
            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, round(fps), (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("영상 끝 또는 손상된 프레임입니다.");
                    break;

                # 추론
                results = self.model(frame)

                # 결과 시각화
                annotated_frame = frame.copy();
                
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0]);
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2;
                    if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0 :
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2);
                
                # ROI 그리기
                cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2);

                out.write(annotated_frame)

            cap.release()
            out.release()

        # 이미지일 경우
        else:
            # 이미지 추론
            img = cv2.imread(input_path)
            results = self.model(img)
            annotated = results[0].plot()
            cv2.imwrite(output_path, annotated)
