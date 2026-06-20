import streamlit as st
import cv2
import tempfile
import os
from src.detector import CrowdPaulse

st.set_page_config(page_title="시장 혼잡도 분석 시스템", layout="wide")
st.title("📊 YOLOv8 + ByteTrack / BoT-SORT 기반 객체 추적 및 혼잡도 분석")

# if "detector" not in st.session_state:
st.session_state.detector = CrowdPaulse()
detector = st.session_state.detector

# ================= [사이드바 레이아웃] =================
st.sidebar.header("📁 데이터 입력")
uploaded_file = st.sidebar.file_uploader("분석할 시장 영상(MP4)을 업로드하세요.", type=["mp4", "avi", "mov"])

st.sidebar.markdown("---")

st.sidebar.header("⚙️ 시스템 설정")
tracker_type = st.sidebar.selectbox("추적 알고리즘 선택", ["ByteTrack (모션 중심)", "BoT-SORT (모션 + 외형 Re-ID)"])

st.sidebar.markdown("---")

st.sidebar.header("📺 시각화 옵션")
show_id = st.sidebar.checkbox("객체 ID 표시", value=True)
show_heatmap = st.sidebar.checkbox("🔥 실시간 혼잡도 히트맵(Heatmap) 하단 생성", value=False)

st.sidebar.markdown("---")
# =======================================================

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    cap = cv2.VideoCapture(video_path)
    
    # 📌 [1층 레이아웃 선언] 상단 결과 배치 (좌우 1:1 분할)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📹 원본 입력 영상 (Original Source)")
        src_window = st.image([]) 
        
    with col2:
        st.subheader("🎯 실시간 객체 추적 (Object Tracking)")
        frame_window = st.image([]) 
        
    st.markdown("---")
    
    # 📌 [2층 레이아웃 선언] 중단 수치 메트릭 공간 생성
    metric_area = st.empty()  # 루프 내에서 실시간으로 수치와 혼잡도 라벨을 갈아 끼울 빈 컨테이너
    
    st.markdown("---")

    # 📌 [3층 레이아웃 선언] 가장 하단에 추적결과와 히트맵이 같이 나오는 공간 구성
    bottom_window_1 = None
    bottom_window_2 = None
    
    if show_heatmap:
        # 하단 공간을 다시 좌우 1:1로 분할하여 추적 영상과 히트맵을 매핑시킵니다.
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            st.subheader("📋 모니터링 참조 스트림 (Tracking Base)")
            bottom_window_1 = st.image([])
        with b_col2:
            st.subheader("🔥 실시간 영역별 혼잡도 차트 (Spatial Heatmap)")
            bottom_window_2 = st.image([])

    # 분석 시작 버튼
    if st.sidebar.button("▶ 분석 시작"):
        if hasattr(detector, 'heatmap_obj'):
            detector.heatmap_obj = None

        # --------------------------------------------------------------------------
        # 📌 [4층 - 핵심 지표 레이아웃] 버튼 누르자마자 루프 시작 전에 미리 뼈대 그려두기
        # --------------------------------------------------------------------------
        st.markdown("---")
        st.header("📉 추적 알고리즘 성능 평가 지표 (Benchmark Metrics)")
        st.caption(f"현재 선택된 [{tracker_type}] 알고리즘의 전통시장 데이터셋 기준 핵심 성능 스코어보드입니다.")

        TRACKER_BENCHMARK = {
            "ByteTrack (모션 중심)": {
                "mAP50": "70.7%", "mAP50_95": "43.05%", "latency": "12.7 ms",
                "mota": "62.3%", "idf1": "67.2%", "id_switch": "35회"
            },
            "BoT-SORT (모션 + 외형 Re-ID)": {
                "mAP50": "70.7%", "mAP50_95": "43.05%", "latency": "12.7 ms",
                "mota": "66.5%", "idf1": "83.5%", "id_switch": "8회"
            }
        }

        metrics = TRACKER_BENCHMARK[tracker_type]

        # 1열: 다중 객체 추적(MOT) 핵심 지표 배치
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="🎯 MOTA (추적 정확도)", value=metrics["mota"])
        with col2:
            st.metric(label="🆔 IDF1 (ID 식별 일치도)", value=metrics["idf1"])
        with col3:
            st.metric(label="🔄 ID Switch (오클루전 누적)", value=metrics["id_switch"], delta="낮을수록 우수", delta_color="inverse")

        st.write("") 

        # 2열: 객체 탐지 및 시스템 효율성 지표 배치
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric(label="🖼️ mAP50 / mAP50-95", value=f"{metrics['mAP50']} / {metrics['mAP50_95']}")
        with col5:
            st.metric(label="⏱️ Inference Latency (추론 속도)", value=metrics["latency"])
       
            
        st.markdown("---") # 지표 아래 깔끔한 마감선
            
        # --------------------------------------------------------------------------
        # 📌 [실시간 영상 추적 루프] 지표를 먼저 그려놓고 그 밑에서 루프를 돌립니다.
        # --------------------------------------------------------------------------
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            original_frame = frame.copy()
                
            # 백엔드 엔진 연산 호출
            annotated_frame, heatmap_frame, current_count = detector.track_frame(
                frame=frame, tracker_type=tracker_type, show_id=show_id, show_heatmap=show_heatmap
            )
            
            # [1층 - 실시간 드로잉] 상단 결과 영상 출력
            rgb_orig = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            src_window.image(rgb_orig, channels="RGB", use_container_width=True)
            
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(rgb_frame, channels="RGB", use_container_width=True)
            
            # [3층 - 실시간 드로잉] 가장 하단 결과영상 + 히트맵 세트 출력
            if show_heatmap and bottom_window_1 is not None and bottom_window_2 is not None:
                bottom_window_1.image(rgb_frame, channels="RGB", use_container_width=True)
                rgb_heatmap = cv2.cvtColor(heatmap_frame, cv2.COLOR_BGR2RGB)
                bottom_window_2.image(rgb_heatmap, channels="RGB", use_container_width=True)
                
            # [2층 - 실시간 드로잉] 중단 인원수 업데이트
            if current_count <= 2:
                status_text = "여유"
                status_color = "🟢"
                status_style = "background-color: #28a745; color: white;" 
            elif current_count <= 4:
                status_text = "보통"
                status_color = "🟡"
                status_style = "background-color: #ffc107; color: black;" 
            else:
                status_text = "혼잡"
                status_color = "🔴"
                status_style = "background-color: #dc3545; color: white;" 

            with metric_area.container():
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric(label="👥 현재 구역 내 실시간 인원수", value=f"{current_count} 명")
                with m_col2:
                    st.markdown("<p style='margin-bottom: 5px; font-size: 14px; color: #666;'>⚠️ 실시간 혼잡도 등급</p>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='{status_style} padding: 8px 16px; border-radius: 5px; "
                        f"font-weight: bold; text-align: center; font-size: 18px; width: 120px;'>"
                        f"{status_color} {status_text}"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
            
        cap.release()
        st.success("🎉 영상 분석이 완료되었습니다!")
        os.unlink(video_path)

else:
    st.info("💡 왼쪽 사이드바에서 시장 비디오 파일을 업로드한 뒤 옵션을 선택하고 [분석 시작] 버튼을 눌러주세요.")