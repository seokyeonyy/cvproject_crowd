import streamlit as st
import cv2
import tempfile
import os
import time
from src.detector import CrowdPulse

st.set_page_config(page_title="시장 혼잡도 분석 시스템", layout="wide")
st.title("📊 YOLOv8 + ByteTrack / BoT-SORT 기반 객체 추적 및 혼잡도 분석")

if "detector" not in st.session_state:
    st.session_state.detector = CrowdPulse()
detector = st.session_state.detector

# ================= [사이드바 레이아웃] =================
st.sidebar.header("📁 데이터 입력")
uploaded_file = st.sidebar.file_uploader("분석할 시장 영상(MP4)을 업로드하세요.", type=["mp4", "avi", "mov"])
st.sidebar.markdown("---")
st.sidebar.header("⚙️ 시스템 설정")
tracker_type = st.sidebar.selectbox("추적 알고리즘 선택", ["ByteTrack (모션 중심)", "BoT-SORT (모션 + 외형 Re-ID)"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
st.sidebar.markdown("---")
st.sidebar.header("📺 시각화 옵션")
show_id = st.sidebar.checkbox("객체 ID 표시", value=True)
show_heatmap = st.sidebar.checkbox("🔥 실시간 혼잡도 히트맵(Heatmap) 하단 생성", value=False)
st.sidebar.markdown("---")

# ================= [메인 로직] =================
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    cap = cv2.VideoCapture(video_path)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📹 원본 입력 영상")
        src_window = st.image([]) 
    with col2:
        st.subheader("🎯 실시간 객체 추적")
        frame_window = st.image([]) 
        
    st.markdown("---")
    metric_area = st.empty() 
    st.markdown("---")

    bottom_window_1, bottom_window_2 = None, None
    if show_heatmap:
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            st.subheader("📋 모니터링 참조 스트림")
            bottom_window_1 = st.image([])
        with b_col2:
            st.subheader("🔥 실시간 영역별 혼잡도 차트")
            bottom_window_2 = st.image([])

    if st.sidebar.button("▶ 분석 시작"):
        # 📌 [성능 평가 지표 출력]
        st.markdown("---")
        st.header("📉 추적 알고리즘 성능 평가 지표 (Benchmark Metrics)")
        st.caption(f"현재 선택된 [{tracker_type}] 알고리즘의 전통시장 데이터셋 기준 핵심 성능 스코어보드입니다.")
        
        TRACKER_BENCHMARK = {
            "ByteTrack (모션 중심)": {"mAP50": "70.7%", "mAP50_95": "43.05%", "latency": "12.7 ms", "mota": "62.3%", "idf1": "67.2%", "id_switch": "35회"},
            "BoT-SORT (모션 + 외형 Re-ID)": {"mAP50": "70.7%", "mAP50_95": "43.05%", "latency": "12.7 ms", "mota": "66.5%", "idf1": "83.5%", "id_switch": "8회"}
        }
        metrics = TRACKER_BENCHMARK[tracker_type]

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("🎯 MOTA (추적 정확도)", metrics["mota"])
        m_col2.metric("🆔 IDF1 (ID 식별 일치도)", metrics["idf1"])
        m_col3.metric("🔄 ID Switch (오클루전)", metrics["id_switch"], delta="낮을수록 우수", delta_color="inverse")

        m_col4, m_col5, m_col6 = st.columns(3)
        m_col4.metric("🖼️ mAP50 / mAP50-95", f"{metrics['mAP50']} / {metrics['mAP50_95']}")
        m_col5.metric("⏱️ Inference Latency", metrics["latency"])
        st.markdown("---")

        # 실시간 추적 루프
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            annotated_frame, heatmap_frame, current_count = detector.track_frame(
                frame=frame, tracker_type=tracker_type, conf_threshold=conf_threshold, 
                show_id=show_id, show_heatmap=show_heatmap
            )
            
            src_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(rgb_frame, channels="RGB", use_container_width=True)
            
            if show_heatmap and bottom_window_1:
                bottom_window_1.image(rgb_frame, channels="RGB", use_container_width=True)
                bottom_window_2.image(cv2.cvtColor(heatmap_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            # [인원수와 혼잡도를 가로로 배치]
            status_text = "여유" if current_count <= 2 else ("보통" if current_count <= 4 else "혼잡")
            status_color = "🟢" if current_count <= 2 else ("🟡" if current_count <= 4 else "🔴")
            
            with metric_area.container():
                cols = st.columns([1, 1])
                cols[0].metric(label="👥 현재 실시간 인원수", value=f"{current_count} 명")
                cols[1].markdown(f"**⚠️ 실시간 혼잡도 등급**")
                cols[1].markdown(f"### {status_color} {status_text}")
        
        cap.release()
        time.sleep(0.5)
        if os.path.exists(video_path): os.unlink(video_path)
        st.success("🎉 분석 완료!")