from ultralytics import YOLO
import cv2
import numpy as np

class YOLOModel:
    def __init__(self):
        self.model = YOLO("./weights/yolo/yolov8l.pt")

    def predict(self, input_path, output_path, polygon_data=None):
        # 입력 파일 확장자 확인
        is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov'))

        points = [];
        data = {};

        # polygon 좌표 numpy 배열로 변환
        if (polygon_data) :
            for pts in polygon_data :
                pts = np.array(
                    [[int(p.x), int(p.y)] for p in pts.points],
                    dtype=np.int32
                );
                points.append(pts);

        # 영상일 경우
        if is_video:
            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, round(fps), (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("영상 끝 또는 손상된 프레임입니다.");
                    break;

                # 추론
                results = self.model(frame)

                # 결과 시각화
                annotated_frame = None;
                
                if (polygon_data) :
                    for pts in points :
                        annotated_frame = frame.copy();
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0]);
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2;
                            if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0 :
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3);
                    
                        # ROI 그리기
                        cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=4);
                else :
                    annotated_frame = results[0].plot()
                    

                out.write(annotated_frame)

            cap.release()
            out.release()

        # 이미지일 경우
        else:
            # 이미지 추론
            img = cv2.imread(input_path)
            results = self.model(img)

            # 결과 시각화
            annotated_frame = None;
                
            if (polygon_data) :
                for pts in points :
                    annotated_frame = img.copy();
                    for box in results[0].boxes:
                        # 바운딩 박스 좌표
                        x1, y1, x2, y2 = map(int, box.xyxy[0]);
                        # 텍스트 (라벨)
                        class_id = int(box.cls[0])
                        label = results[0].names[class_id]
                        text = f"{label}"
                        # 정확도
                        conf = float(box.conf[0]);
                        data[label] = conf;

                        # 바운딩 박스 중심점
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2;
                        if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0 :
                            # ROI 내부 바운딩 박스 그리기
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3);
                            
                            # 라벨 배경
                            (text_width, text_height), baseline = cv2.getTextSize(
                                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                            )
                            top_left = (x1, y1 - text_height - 8)
                            bottom_right = (x1 + text_width + 4, y1)
                            cv2.rectangle(annotated_frame, top_left, bottom_right, (0, 0, 255), thickness=-1)

                            # 라벨
                            cv2.putText(
                                annotated_frame,
                                text,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA
                            )
                    
                    # ROI 그리기
                    # cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=4);
            else :
                annotated_frame = results[0].plot()
            cv2.imwrite(output_path, annotated_frame)

        if polygon_data :
            return data;
        else :
            return None;
