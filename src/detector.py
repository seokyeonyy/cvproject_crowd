import os
import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st

class CrowdPaulse:
    def __init__(self, model_path=None):
        # 1. 파일 위치 기반 절대 경로 계산
        # 현재 파일(detector.py) 위치: .../project_root/src/detector.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 프로젝트 최상위 루트: .../project_root/
        project_root = os.path.dirname(current_dir)
        
        # 2. 경로 조합
        # weights 폴더가 프로젝트 루트 바로 아래에 있으므로 이 경로가 맞습니다.
        self.model_path = os.path.join(project_root, "weights", "best.pt")
        
        # 3. 파일 존재 여부 확인 (디버깅용)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
            
        self.model = YOLO(self.model_path)
        self.accumulated_heatmap = None
        
    # 기존 코드의 track_frame 정의부를 아래와 같이 변경하세요.
    def track_frame(self, frame, tracker_type, conf_threshold, show_id=True, show_heatmap=False):
        # [방어적 코드] 입력 데이터 유효성 검증
        if self.model is None or frame is None or frame.size == 0:
            return frame, frame, 0

        # [전처리] 채널 확인 및 변환
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        annotated_frame = frame.copy()
        heatmap_frame = frame.copy()
        current_count = 0

        if self.accumulated_heatmap is None or self.accumulated_heatmap.shape != frame.shape[:2]:
            self.accumulated_heatmap = np.zeros(frame.shape[:2], dtype=np.float32)

        tracker_name = "bytetrack.yaml" if "ByteTrack" in tracker_type else "botsort.yaml"

        try:
            # 💡 여기서 app.py에서 전달받은 conf_threshold를 사용합니다!
            results = self.model.track(
                source=frame, 
                tracker=tracker_name, 
                persist=True, 
                conf=conf_threshold,  # 외부에서 받은 값 적용
                iou=0.45, 
                verbose=False
            )

            # (이하 결과 처리 및 시각화 로직은 기존과 동일)
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                scores = results[0].boxes.conf.cpu().numpy()
                current_count = len(ids)

  
                # 히트맵 연산
                if show_heatmap:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        point_canvas = np.zeros(frame.shape[:2], dtype=np.uint8)
                        cv2.circle(point_canvas, (center_x, center_y), 60, 255, -1)
                        blur_point = cv2.GaussianBlur(point_canvas, (51, 51), 0)
                        add_value = (blur_point * 0.05).astype(np.float32)
                        self.accumulated_heatmap = cv2.add(self.accumulated_heatmap, add_value)

                # 바운딩 박스 & ID 시각화
                for box, obj_id, score in zip(boxes, ids, scores):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    if show_id:
                        text = f"ID:{obj_id} {score:.2f}"
                        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 3)
                        cv2.rectangle(annotated_frame, (x1, y1 - text_h - 14), (x1 + text_w + 14, y1), (0, 0, 255), -1)
                        cv2.putText(annotated_frame, text, (x1 + 7, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 3, cv2.LINE_AA)

                # 컬러맵 합성
                if show_heatmap:
                    display_heat = cv2.normalize(self.accumulated_heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    color_map = cv2.COLORMAP_SUMMER if current_count <= 2 else (cv2.COLORMAP_FALL if current_count <= 4 else cv2.COLORMAP_HOT)
                    color_mapped = cv2.applyColorMap(display_heat, color_map)
                    _, mask = cv2.threshold(display_heat, 10, 255, cv2.THRESH_BINARY)
                    heatmap_frame = cv2.addWeighted(cv2.bitwise_and(color_mapped, color_mapped, mask=mask), 0.7, frame, 0.3, 0)

        except Exception as e:
            print(f"[Model Inference Error] {e}")
            return frame, frame, 0

        return annotated_frame, heatmap_frame, current_count