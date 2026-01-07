"""
리뷰 분석 Streamlit 웹 서비스

화면 흐름:
1. 메인 화면 (서비스 소개 + 시작 버튼)
2. 업로드 화면 (엑셀 업로드)
3. 대시보드 화면 (AI 분석 결과 시각화)

※ Streamlit에는 진짜 페이지 이동이 없기 때문에
   session_state.page 값을 바꿔서 화면을 전환한다.
"""

# ==============================
# 1. 라이브러리 import
# ==============================
import os
import json
import re
import platform

# =========================
# Streamlit & 데이터 처리
# =========================
import streamlit as st
import pandas as pd

# =========================
# OpenAI (GPT)
# =========================
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# 시각화 (대시보드)
# =========================
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ==============================
# 2. 환경 변수 로드
# ==============================
load_dotenv()

# ==============================
# 3. OpenAI 클라이언트 생성
# ==============================
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ==============================
# 4. 앱 기본 설정
# ==============================
st.set_page_config(
    page_title="리뷰 분석 서비스",
    page_icon="📊",
    layout="wide"
)

# ==============================
# 5. Session State 초기화
# ==============================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "result" not in st.session_state:
    st.session_state.result = None

# ==============================
# 6. 공통 CSS
# ==============================
st.markdown("""
<style>
.metric-card {
    padding: 20px;
    border-radius: 14px;
    background-color: white;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    text-align: center;
}
.section-gap {
    margin-top: 32px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# 7. AI 분석 로직
# ==============================
def analyze_reviews(df: pd.DataFrame):
    """
    CSV로 업로드된 리뷰 데이터를 GPT로 분석
    """

    if "review" not in df.columns:
        return {
            "total": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "score": 0.0,
            "keywords": [],
            "summary": ""
        }

    reviews = (
        df["review"]
        .dropna()
        .astype(str)
        .tolist()
    )

    if not reviews:
        return {
            "total": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "score": 0.0,
            "keywords": [],
            "summary": ""
        }

    prompt = f"""
아래는 고객 리뷰 목록입니다.

리뷰:
{chr(10).join(reviews[:50])}

이 리뷰들을 분석해서 반드시 아래 JSON 형식으로만 답변하세요.

{{
  "total": 전체 리뷰 수 (정수),
  "positive": 긍정 리뷰 수 (정수),
  "neutral": 중립 리뷰 수 (정수),
  "negative": 부정 리뷰 수 (정수),
  "score": 전체 리뷰 만족도를 0~10점 사이 숫자로 평가 (소수점 1자리),
  "keywords": ["형태소 기준 핵심 키워드 5개"],
  "summary": "전체 리뷰 요약 문단"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 고객 리뷰 분석 전문가다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        content = response.choices[0].message.content
        match = re.search(r"\{.*\}", content, re.DOTALL)
        gpt_result = json.loads(match.group())

    except Exception:
        return {
            "total": len(reviews),
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "score": 0.0,
            "keywords": [],
            "summary": ""
        }

    return {
        "total": int(gpt_result.get("total", len(reviews))),
        "positive": int(gpt_result.get("positive", 0)),
        "neutral": int(gpt_result.get("neutral", 0)),
        "negative": int(gpt_result.get("negative", 0)),
        "score": float(gpt_result.get("score", 0.0)),
        "keywords": gpt_result.get("keywords", []),
        "summary": gpt_result.get("summary", "")
    }


# ==============================
# 8. 메인 화면
# ==============================
def render_home():
    st.markdown("""
    <h1 style="font-size:48px;">📊 리뷰 분석 서비스</h1>
    <p style="font-size:18px; color:#6B7280;">
    엑셀 리뷰 데이터를 업로드하면<br>
    AI가 자동으로 핵심 인사이트를 도출합니다.
    </p>
    """, unsafe_allow_html=True)

    if st.button("🚀 리뷰 분석 시작", use_container_width=True):
        st.session_state.page = "upload"
        st.rerun()


# ==============================
# 9. 업로드 화면
# ==============================
def render_upload():
    st.title("📂 리뷰 데이터 업로드")

    uploaded_file = st.file_uploader(
        " ",
        type=["csv", "xlsx"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="cp949")

        st.dataframe(df.head())

        if st.button("🤖 AI 분석 실행", use_container_width=True):
            with st.spinner("AI 분석 중..."):
                st.session_state.result = analyze_reviews(df)
            st.session_state.page = "dashboard"
            st.rerun()

    if st.button("← 메인으로"):
        st.session_state.page = "home"
        st.rerun()


# ==============================
# 10. 대시보드 화면
# ==============================
def render_dashboard():
    st.title("📊 리뷰 분석 대시보드")

    result = st.session_state.get("result")
    if not result:
        st.warning("분석 결과가 없습니다.")
        return

    # KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 리뷰 수", result["total"])
    c2.metric("긍정 😊", result["positive"])
    c3.metric("중립 😐", result["neutral"])
    c4.metric("부정 😡", result["negative"])

    # 감성 데이터
    sentiment_df = pd.DataFrame({
        "감성": ["긍정", "중립", "부정"],
        "리뷰 수": [
            result["positive"],
            result["neutral"],
            result["negative"]
        ]
    }).set_index("감성")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 감성 분포")
        st.bar_chart(sentiment_df)

    with col2:
        st.subheader("🥧 감성 비율")

        # ===== 폰트 안전 처리 =====
        plt.rcParams["axes.unicode_minus"] = False
        font_prop = None

        if platform.system() == "Windows":
            font_path = "C:/Windows/Fonts/malgun.ttf"
            if os.path.exists(font_path):
                font_prop = fm.FontProperties(fname=font_path)
                plt.rcParams["font.family"] = font_prop.get_name()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            sentiment_df["리뷰 수"],
            labels=None,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.legend(
            sentiment_df.index,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            prop=font_prop if font_prop else None
        )

        ax.set_title("감성 비율", fontproperties=font_prop if font_prop else None)
        st.pyplot(fig)

    # 키워드
    st.subheader("🔑 주요 키워드")
    cols = st.columns(len(result["keywords"]))
    for c, k in zip(cols, result["keywords"]):
        c.metric(k, "")

    # 점수
    st.subheader("⭐ 종합 만족도")
    st.markdown(f"## {result['score']} / 10")

    # 요약
    st.subheader("📝 AI 요약")
    st.write(result["summary"])

    if st.button("🏠 메인으로"):
        st.session_state.page = "home"
        st.rerun()


# ==============================
# 11. 화면 라우팅
# ==============================
if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "upload":
    render_upload()
elif st.session_state.page == "dashboard":
    render_dashboard()
