import os
import cv2
import numpy as np
from ultralytics import YOLO

class CrowdPaulse:
   def __init__(self):
    # 현재 src/detector.py 파일의 위치 기준 디렉토리 파악
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # 프로젝트 최상위 루트

    # 1. 가중치 파일 경로를 프로젝트 루트 기준으로 명확히 설정
    self.model_path = os.path.join(project_root, "weights", "best.pt") 
    self.model = YOLO(self.model_path)

    # 2. YAML 설정 파일 경로 (기존 코드 유지)
    self.bytetrack_yaml = os.path.join(project_root, "configs", "bytetrack.yaml")
    self.botsort_yaml = os.path.join(project_root, "configs", "botsort.yaml")

    self.accumulated_heatmap = None

   def track_frame(self, frame, tracker_type, show_id=True, show_heatmap=False):
        annotated_frame = frame.copy()
        heatmap_frame = frame.copy() # 기본값은 원본 영상
        current_count = 0

        # 누적 도화지 크기 자동 설정 (영상 크기와 동일하게)
        if self.accumulated_heatmap is None or self.accumulated_heatmap.shape != frame.shape[:2]:
            self.accumulated_heatmap = np.zeros(frame.shape[:2], dtype=np.float32)

        # 트래커 설정 선택
        if "ByteTrack" in tracker_type:
            tracker_name = "bytetrack.yaml"
        else:
            tracker_name= "botsort.yaml"
            
        results = self.model.track(
            source=frame, tracker=tracker_name, persist=True, conf=0.15, iou=0.45, verbose=False
        )
        
        # 탐지된 객체 데이터가 있을 때 로직 시작
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            scores = results[0].boxes.conf.cpu().numpy()
            
            current_count = len(ids)
            
            # -------------------------------------------------------------
            # 💡 [구현 1 & 2] 좌표값 강제 주입 구조 & 부드러운 불꽃 효과 연산
            # -------------------------------------------------------------
            if show_heatmap:
                # 매 프레임 탐지된 사람들의 위치에 반지름 60의 가우시안 불꽃 소스를 누적합니다.
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    # 바운딩 박스의 하단 중심점(발 위치) 또는 중앙점 계산
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # 1. np.zeros 기반 임시 캔버스 생성
                    point_canvas = np.zeros(frame.shape[:2], dtype=np.uint8)
                    # 2. 중심점에 cv2.circle로 흰색 원 직접 주입 (반지름 60)
                    cv2.circle(point_canvas, (center_x, center_y), 60, 255, -1)
                    # 3. 가우시안 블러를 먹여 테두리가 흐릿하고 부드러운 불꽃 형태로 가공
                    blur_point = cv2.GaussianBlur(point_canvas, (51, 51), 0)
                    
                    # 4. 가공된 불꽃 강도를 메인 누적 도화지에 더해줌 (지속적으로 밟은 곳이 붉어지도록)
                    # 값을 0.05 정도로 연하게 주어 누적 효과를 극대화합니다.
                    add_value = (blur_point * 0.05).astype(np.float32)
                    self.accumulated_heatmap = cv2.add(self.accumulated_heatmap, add_value)

            # [바운딩 박스 & ID 시각화 레이어]
            for box, obj_id, score in zip(boxes, ids, scores):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                
                if show_id:
                    text = f"ID:{obj_id} person {score:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.95
                    thickness = 3
                    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    label_y1 = max(y1 - text_h - 14, 0)
                    label_y2 = label_y1 + text_h + 14
                    cv2.rectangle(annotated_frame, (x1, label_y1), (x1 + text_w + 14, label_y2), color, -1)
                    cv2.putText(annotated_frame, text, (x1 + 7, label_y2 - 7 - baseline), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 💡 [구현 3 & 4] 실시간 동적 컬러맵 연동 및 정교한 알파 블렌딩 마스킹
        # -------------------------------------------------------------
        if show_heatmap:
            try:
                # 1. 현재 누적된 강도 데이터를 0~255로 정규화
                # 이 과정을 거쳐야 값이 작아도 무조건 화면에 나타납니다.
                display_heat = cv2.normalize(self.accumulated_heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # 2. 인원 밀도 기반의 동적 컬러맵 결정 (여유: 초록 테마, 보통: 가을/노랑 테마, 혼잡: 붉은 테마)
                if current_count <= 2:
                    # 🟢 여유 상태: 초록-파랑 계열 (WINTER나 SUMMER 컬러맵 활용)
                    color_mapped = cv2.applyColorMap(display_heat, cv2.COLORMAP_SUMMER)
                elif current_count <= 4:
                    # 🟡 보통 상태: 노랑-주황 계열 (기존 FALL 컬러맵 활용)
                    color_mapped = cv2.applyColorMap(display_heat, cv2.COLORMAP_FALL)
                else:
                    # 🔴 혼잡 상태: 완전히 새빨간 불꽃 계열 (JET 혹은 HOT 컬러맵 활용)
                    color_mapped = cv2.applyColorMap(display_heat, cv2.COLORMAP_HOT)

                # 3. 투박한 사각형 덮어쓰기 방지를 위한 정교한 마스킹 연산
                # 불꽃 값이 거의 없는 부분(임계값 10 이하)은 투명하게 처리하기 위해 마스크 생성
                _, mask = cv2.threshold(display_heat, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                
                # 배경(원본 프레임에서 히트맵이 없는 부분 추출)
                bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                # 전경(컬러맵 이미지에서 히트맵이 존재하는 부분만 추출)
                fg = cv2.bitwise_and(color_mapped, color_mapped, mask=mask)
                
                # 4. 최종 알파 블렌딩 합성 (히트맵 70% + 원본 30% 혼합하여 쨍하게 노출)
                heatmap_base = cv2.add(bg, fg)
                heatmap_frame = cv2.addWeighted(heatmap_base, 0.7, frame, 0.3, 0)

            except Exception as e:
                print(f"[Heatmap Render Error] {e}")
                heatmap_frame = frame.copy()
        else:
            # 히트맵 기능이 꺼져있을 때는 매끄러운 처리를 위해 누적 도화지 초기화
            self.accumulated_heatmap = None
            heatmap_frame = frame.copy()
                                
        return annotated_frame, heatmap_frame, current_count