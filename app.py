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

# 프로젝트 루트의 .env 파일을 읽어서
# OPENAI_API_KEY 같은 환경 변수를 시스템에 등록
load_dotenv()


# ==============================
# 2. OpenAI 클라이언트 생성
# ==============================

# .env에 저장된 OPENAI_API_KEY를 불러와
# OpenAI API와 통신하기 위한 클라이언트 객체 생성
#
# 이 client 객체를 통해 GPT 모델을 호출하게 된다.
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ==============================
# 3. 앱 기본 설정 (필수)
# ==============================
st.set_page_config(
    page_title="리뷰 분석 서비스",  # 브라우저 탭 제목
    page_icon="📊",                # 파비콘
    layout="wide"                  # 대시보드용 넓은 레이아웃
)

# ==============================
# 4. Session State 초기화
# ==============================
# 현재 보고 있는 화면 상태
if "page" not in st.session_state:
    st.session_state.page = "home"   # home | upload | dashboard

# 분석 결과 저장용
if "result" not in st.session_state:
    st.session_state.result = None

# ==============================
# 5. 공통 CSS (카드 스타일)
# ==============================
# Streamlit 기본 UI는 밋밋하므로
# 카드 느낌을 주기 위한 최소한의 CSS만 사용
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


def extract_review_texts(df: pd.DataFrame) -> list[str]:
    """
    설문 응답 DataFrame에서
    '응답자 1명 = 리뷰 1개' 기준으로
    리뷰 텍스트를 추출
    """

    reviews = []

    for _, row in df.iterrows():
        texts = []

        for value in row.values:
            if pd.isna(value):
                continue

            value = str(value).strip()

            # 숫자만 있는 값 제외 (만족도 점수 등)
            if value.replace(".", "").isdigit():
                continue

            # 너무 짧은 텍스트 제외
            if len(value) < 5:
                continue

            texts.append(value)

        # 한 행의 텍스트를 하나로 합침
        if texts:
            combined = " / ".join(texts)
            reviews.append(combined)

    return reviews


# ==============================
# 6. 비즈니스 로직 (AI 분석 영역)
# ==============================
def analyze_reviews(reviews: list[str]):
    """
    다국어(한국어/영어/혼합) 리뷰 리스트를 입력받아
    분석 결과는 무조건 한국어로 반환한다.

    - 리뷰 개수/긍정/중립/부정 계산은 Python에서 수행
    - GPT는 감성 판단과 요약만 담당
    """

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

    sample_reviews = reviews[:50]

    prompt = f"""
아래는 고객 설문 및 리뷰 응답 목록입니다.
응답은 한국어, 영어 또는 혼합 언어일 수 있습니다.

리뷰 목록:
{chr(10).join(sample_reviews)}

각 리뷰에 대해 감성을 판단하세요.

규칙:
- 각 리뷰마다 하나의 감성만 선택
- 선택지는 반드시 아래 중 하나:
  - positive
  - neutral
  - negative
- 개수나 통계는 계산하지 말 것
- 모든 설명과 요약은 한국어로 작성
- 키워드는 원문 언어를 유지

반드시 아래 JSON 형식으로만 답변하세요.

{{
  "sentiments": ["positive", "neutral", "negative", ...],
  "score": 전체 만족도를 0~10점 사이 숫자로 평가 (소수점 1자리),
  "keywords": ["핵심 키워드 5개"],
  "summary": "전체 리뷰를 한 문단으로 요약한 한국어 문장"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 다국어 설문 데이터를 분석하는 전문가다. "
                        "입력 언어와 관계없이 분석 결과는 반드시 한국어로 제공해야 한다."
                    )
                },
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

    # =========================
    # Python에서 감성 집계
    # =========================
    sentiments = gpt_result.get("sentiments", [])

    # 안전 장치 (길이 불일치 방어)
    sentiments = sentiments[:len(reviews)]

    total = len(reviews)
    positive = sentiments.count("positive")
    neutral = sentiments.count("neutral")
    negative = sentiments.count("negative")

    return {
        "total": total,
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "score": float(gpt_result.get("score", 0.0) or 0.0),
        "keywords": gpt_result.get("keywords", []) or [],
        "summary": gpt_result.get("summary", "") or ""
    }


# ==============================
# 7-1. 메인 화면 (랜딩 페이지)
# ==============================
def render_home():
    """
    서비스 소개용 메인 화면
    - 서비스 설명
    - 기능 요약
    - '리뷰 분석 시작' CTA 버튼
    """

    # 제목 + 설명 (히어로 영역)
    st.markdown("""
    <h1 style="font-size:48px;">📊 리뷰 분석 서비스</h1>
    <p style="font-size:18px; color:#6B7280;">
    엑셀 리뷰 데이터를 업로드하면<br>
    AI가 자동으로 핵심 인사이트를 도출합니다.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # CTA 버튼 (가장 중요)
    if st.button("🚀 리뷰 분석 시작", use_container_width=True):
        st.session_state.page = "upload"
        st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    # 기능 요약 카드 3개
    col1, col2, col3 = st.columns(3)

    col1.markdown("""
    <div class="metric-card">
        <h3>📂 엑셀 업로드</h3>
        <p style="color:#6B7280;">
        리뷰 데이터를 한 번에 업로드
        </p>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown("""
    <div class="metric-card">
        <h3>🤖 AI 분석</h3>
        <p style="color:#6B7280;">
        감성·키워드 자동 분석
        </p>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown("""
    <div class="metric-card">
        <h3>📈 대시보드</h3>
        <p style="color:#6B7280;">
        한 눈에 보는 인사이트
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

    # 사용 방법 안내
    st.markdown("### 사용 방법")
    step1, step2, step3 = st.columns(3)

    step1.markdown("**1️⃣ 엑셀 업로드**  \n리뷰 데이터 준비")
    step2.markdown("**2️⃣ AI 분석 실행**  \n자동 인사이트 도출")
    step3.markdown("**3️⃣ 결과 확인**  \n대시보드에서 확인")

# ==============================
# 7-2. 업로드 화면
# ==============================
def render_upload():
    """
    엑셀 파일 업로드 화면
    - 파일 업로드
    - 데이터 미리보기
    - 분석 실행 버튼
    """

    st.title("📂 리뷰 데이터 업로드")
    st.caption("엑셀(.xlsx) 파일을 업로드하세요")

    st.divider()

    uploaded_file = st.file_uploader(
        " ",
        type=["csv", "xlsx"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        
        file_name = uploaded_file.name.lower()

        try:
            if file_name.endswith(".csv"):
                try:
                    df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
                except UnicodeDecodeError:
                    df = pd.read_csv(uploaded_file, encoding="cp949")

            elif file_name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)

            else:
                st.error("지원하지 않는 파일 형식입니다.")
                return

        except Exception as e:
            st.error("파일을 읽는 중 오류가 발생했습니다.")
            st.exception(e)
            return
        
        st.success("파일 업로드 완료")
        st.info(f"총 {len(df)}건의 리뷰가 확인되었습니다")

        # 데이터 미리보기
        with st.expander("업로드 데이터 미리보기"):
            st.dataframe(df.head(10), use_container_width=True)

        # 분석 실행 버튼
        if st.button("🤖 AI 분석 실행", use_container_width=True):
            with st.spinner("AI가 리뷰를 분석 중입니다..."):
                reviews = extract_review_texts(df)
                result = analyze_reviews(reviews)

            # 결과 저장 후 대시보드로 이동
            st.session_state.result = result
            st.session_state.page = "dashboard"
            st.rerun()

    st.divider()

    # 메인으로 돌아가기
    if st.button("← 메인으로"):
        st.session_state.page = "home"
        st.rerun()

# ==============================
# 7-3. 대시보드 화면
# ==============================
def render_dashboard():
    """
    GPT가 분석한 리뷰 결과를 시각화하는 대시보드 화면

    포함 요소:
    - KPI 카드 (총 리뷰 / 긍정 / 중립 / 부정)
    - 감성 분포 막대 그래프
    - 감성 비율 파이 차트
    - 주요 키워드 카드
    - 종합 만족도 점수 (10점 만점)
    - GPT 요약 문장
    """

    st.title("📊 리뷰 분석 대시보드")
    st.caption("AI가 분석한 리뷰 인사이트 요약")

    # =========================
    # 1. 분석 결과 가져오기
    # =========================
    result = st.session_state.get("result")

    if not result:
        st.warning("분석 결과가 없습니다. 먼저 리뷰 분석을 실행해주세요.")
        return

    st.divider()

    # =========================
    # 2. KPI 카드 영역
    # =========================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("총 리뷰 수", int(result.get("total", 0)))
    col2.metric("긍정 😊", int(result.get("positive", 0)))
    col3.metric("중립 😐", int(result.get("neutral", 0)))
    col4.metric("부정 😡", int(result.get("negative", 0)))

    st.divider()

    # =========================
    # 3. 감성 데이터프레임 생성
    # =========================
    sentiment_df = pd.DataFrame({
        "감성": ["긍정", "중립", "부정"],
        "리뷰 수": [
            result.get("positive", 0),
            result.get("neutral", 0),
            result.get("negative", 0)
        ]
    }).set_index("감성")

    # =========================
    # 4. 감성 시각화 영역
    # =========================
    col1, col2 = st.columns(2)

    # ---- 4-1. 감성 분포 막대 그래프 ----
    with col1:
        st.subheader("📊 감성 분포 (막대 그래프)")
        st.bar_chart(sentiment_df, use_container_width=True)

    # ---- 4-2. 감성 비율 파이 차트 ----
    with col2:
        st.subheader("🥧 감성 비율 (파이 차트)")

        if sentiment_df["리뷰 수"].sum() == 0:
            st.info("감성 비율을 표시할 데이터가 없습니다.")
        else:
            font_path = os.path.join("assets", "fonts", "malgun.ttf")
            font_prop = fm.FontProperties(fname=font_path)

            plt.rcParams["font.family"] = font_prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False

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
                prop=font_prop
            )

            ax.set_title("감성 비율", fontproperties=font_prop)

            plt.tight_layout()
            st.pyplot(fig)

    st.divider()

    # =========================
    # 5. 주요 키워드 카드 (먼저 표시)
    # =========================
    st.subheader("🔑 주요 키워드")

    keywords = result.get("keywords", [])

    if not keywords:
        st.info("추출된 주요 키워드가 없습니다.")
    else:
        cols = st.columns(min(len(keywords), 6))

        for col, keyword in zip(cols, keywords[:6]):
            col.markdown(
                f"""
                <div style="
                    padding:16px;
                    border-radius:12px;
                    background-color:#f9fafb;
                    text-align:center;
                    font-weight:600;
                ">
                    {keyword}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    # =========================
    # 6. 종합 만족도 점수 (키워드 다음)
    # =========================
    st.subheader("⭐ 종합 만족도")

    score = result.get("score", None)

    if score is not None:
        score = round(float(score), 1)

        if score >= 7:
            bg_color = "#22c55e"
        elif score >= 4:
            bg_color = "#f59e0b"
        else:
            bg_color = "#ef4444"

        st.markdown(
            f"""
            <div style="
                padding:24px;
                border-radius:16px;
                background:{bg_color};
                color:white;
                text-align:center;
                margin-bottom:24px;
            ">
                <div style="font-size:18px; opacity:0.9;">
                    AI 종합 평가
                </div>
                <div style="font-size:48px; font-weight:700;">
                    {score} / 10
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("종합 만족도 점수를 표시할 수 없습니다.")

    st.divider()

    # =========================
    # 7. GPT 요약 문장 (마지막)
    # =========================
    st.subheader("📝 AI 요약")

    summary = result.get("summary", "")

    if summary:
        st.markdown(
            f"""
            <div style="
                padding:20px;
                border-radius:14px;
                background-color:white;
                box-shadow:0 4px 14px rgba(0,0,0,0.06);
                font-size:16px;
                line-height:1.6;
            ">
                {summary}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("요약 문장이 없습니다.")

    st.divider()

    # =========================
    # 8. 하단 네비게이션 버튼
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 새 분석"):
            st.session_state.page = "upload"
            st.rerun()

    with col2:
        if st.button("🏠 메인으로"):
            st.session_state.page = "home"
            st.rerun()


# ==============================
# 8. 화면 라우팅 (Navigation 역할)
# ==============================
if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "upload":
    render_upload()
elif st.session_state.page == "dashboard":
    render_dashboard()

