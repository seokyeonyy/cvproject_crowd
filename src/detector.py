import cv2
from ultralytics import YOLO

class CrowdDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # 초기 구동을 위해 가장 가볍고 자동으로 다운로드되는 yolov8n.pt 모델 사용
        self.model = YOLO(model_path)
        
    def track_frame(self, frame, tracker_type, show_bbox, show_id):
        """
        한 프레임을 받아서 YOLOv8 + Tracker를 돌린 뒤, 
        옵션에 따라 바운딩 박스와 ID를 그린 프레임과 실시간 인원수를 반환합니다.
        """
        # 사용자가 선택한 트래커에 따라 설정 매핑
        # (YOLOv8 내장 트래커인 bytetrack.yaml 또는 botsort.yaml 사용)
        config_name = "bytetrack.yaml" if "ByteTrack" in tracker_type else "botsort.yaml"
        
        # YOLOv8 추적 엔진 구동 (사람 클래스인 'person'은 보통 class 0번입니다)
        results = self.model.track(
            source=frame, 
            persist=True, 
            tracker=config_name, 
            classes=[0],       # 0번(사람)만 탐지 및 추적
            verbose=False      # 터미널 로그 깔끔하게 유지
        )
        
        annotated_frame = frame.copy()
        current_count = 0
        
        # 결과가 존재하고 박스가 잡혔을 때만 시각화 로직 작동
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # 좌표 (x1, y1, x2, y2)
            ids = results[0].boxes.id.cpu().numpy().astype(int)  # 객체 추적 ID
            current_count = len(ids) # 현재 프레임에 감지된 총 사람 수
            
            # 탐지된 사람들을 순회하며 화면에 그림 그리기
            for box, obj_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                
                # 1. Bounding Box 표시 옵션이 켜져 있을 때
                if show_bbox:
                    # 설계도 컨셉에 맞춘 보라색(#7A42F4 -> BGR 기준 (244, 66, 122)) 박스 그리기
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (244, 66, 122), 2)
                
                # 2. ID 표시 옵션이 켜져 있을 때
                if show_id:
                    # 박스 상단에 ID 텍스트 뿌리기
                    label = f"ID: {obj_id}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                
        return annotated_frame, current_count