import streamlit as st
import cv2
import tempfile
import os
import time
from src.detector import CrowdDetector 

# 페이지 기본 설정 (와이드 모드)
st.set_page_config(page_title="시장 혼잡도 분석 시스템", layout="wide")

# 타이틀
st.title("시장 혼잡도 분석 시스템")
st.subheader("YOLOv8 + ByteTrack / BoT-SORT 기반 객체 추적 및 혼잡도 분석")

# 1. 사이드바 (분석 설정 영역)
st.sidebar.title("영상 업로드")
uploaded_file = st.sidebar.file_uploader("파일을 드래그하거나 선택하세요", type=["mp4", "avi", "mov"])

st.sidebar.subheader("Tracker 선택")
tracker_type = st.sidebar.selectbox("기본 트래커 설정", ["ByteTrack (모션 중심)", "BoT-SORT (모션+외형 결합)"])

st.sidebar.subheader("분석 옵션")
show_bbox = st.sidebar.checkbox("Bounding Box 표시", value=True)
show_id = st.sidebar.checkbox("ID 표시", value=True)
show_line = st.sidebar.checkbox("Tracking Line 표시", value=False) # 다음 단계 구현 예정
show_heatmap = st.sidebar.checkbox("Heatmap 생성", value=False)    # 다음 단계 구현 예정

start_btn = st.sidebar.button("▶ 분석 시작", use_container_width=True)

# 2. 메인 화면 - 영상 출력 영역 (2분할 공간 확보)
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📹 원본 영상")
    original_video_spot = st.empty()

with col2:
    st.markdown("### ⚙️ Tracking 결과 영상")
    result_video_spot = st.empty()

# 초기 대기 화면 설정
original_video_spot.image("https://via.placeholder.com/640x360.png?text=Waiting+for+Video...", use_container_width=True)
result_video_spot.image("https://via.placeholder.com/640x360.png?text=Waiting+for+Analysis...", use_container_width=True)

# 3. 메인 화면 - 하단 혼잡도 분석 결과 영역
st.markdown("---")
st.markdown("### 📊 실시간 혼잡도 분석 결과")

metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
with metrics_col1:
    current_count_spot = st.empty()
    current_count_spot.metric(label="현재 인원 수", value="0 명")
with metrics_col2:
    avg_time_spot = st.empty()
    avg_time_spot.metric(label="평균 체류 시간", value="- 초")
with metrics_col3:
    status_spot = st.empty()
    status_spot.metric(label="현재 혼잡도 상태", value="-")

# 4. 분석 시작 핵심 로직 구동
if uploaded_file is not None and start_btn:
    
    # 🆕 탐지기 엔진 초기화 (최초 실행 시 YOLOv8 뼈대 모델을 자동으로 다운로드합니다)
    detector = CrowdDetector()
    
    # 임시 파일 세팅
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # 비디오 루프
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 원본 영상 출력을 위한 RGB 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_video_spot.image(frame_rgb, use_container_width=True)
        
        # 🆕 백엔드 엔진에 한 프레임을 던져서 트래킹 이미지와 인원 수 받아오기
        annotated_frame, current_count = detector.track_frame(
            frame=frame, 
            tracker_type=tracker_type, 
            show_bbox=show_bbox, 
            show_id=show_id
        )
        
        # 결과 영상 출력을 위한 RGB 변환 및 화면 갱신
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        result_video_spot.image(annotated_frame_rgb, use_container_width=True)
        
        # 🆕 대시보드 하단 지표에 실시간 인원 수 매칭하기
        # 기준 인원을 10명으로 잡고 상태 정보 변화 템플릿 추가
        status_text = "혼잡" if current_count >= 10 else "보통" if current_count > 3 else "여유"
        current_count_spot.metric(label="현재 인원 수", value=f"{current_count} 명", delta=f"{current_count - 5}명 기준대비")
        status_spot.metric(label="현재 혼잡도 상태", value=status_text)
        
        # 웹 브라우저가 과부하 걸리지 않게 아주 약간의 휴식 시간 부여
        time.sleep(0.01)
        
    cap.release()
    os.unlink(tfile.name)
    st.success("🎉 영상 분석이 완료되었습니다!")