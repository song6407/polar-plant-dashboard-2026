import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import streamlit as st


# ──────────────────────────────────────────────
# 한글 폰트 & wide mode
# ──────────────────────────────────────────────
st.set_page_config(layout="wide")
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# 데이터 로딩 (한 번만)
# ──────────────────────────────────────────────
@st.cache_data
def read_csv(file):
    """utf-8-sig → cp949 순서로 시도하며 바이트에서 직접 읽는다."""
    if file is None:
        return pd.DataFrame()
    raw = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            return df
        except Exception:
            pass
    return pd.DataFrame()


def std_cols(df):
    """열 이름을 표준 이름으로 바꾸고 숫자 변환을 시도한다."""
    if df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)

    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            extracted = df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0]
            num = pd.to_numeric(extracted, errors="coerce")
            if num.notna().mean() >= 0.5:
                df[c] = num

    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]

    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed

    return df


def school_name_from_filename(filename):
    """업로드한 파일명에서 학교명 추정 (예: 인천송천고_환경데이터.csv → 인천송천고)."""
    if not filename:
        return "우리 학교"
    base = os.path.basename(filename)
    # 확장자 제거
    name = os.path.splitext(base)[0]
    # '_' 또는 ' ' 기준으로 앞부분만 취함
    for sep in ("_", " "):
        if sep in name:
            return name.split(sep)[0]
    return name


# ──────────────────────────────────────────────
# 사이드바: 파일 업로드 & 측정 항목 선택
# ──────────────────────────────────────────────
st.sidebar.title("📁 데이터 업로드")

env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

if env_file is not None or growth_file is not None:
    uploaded_school = school_name_from_filename(
        env_file.name if env_file is not None else (growth_file.name if growth_file is not None else "")
    )
else:
    uploaded_school = "우리 학교"

growth_items = st.sidebar.multiselect(
    "🌱 생육 측정 항목 (그래프에 사용할 항목 선택)",
    ["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량", "생중량"],
    default=["잎 수", "지상부 길이", "지하부 길이", "생중량"],
)


@st.cache_data
def load_data(env_file, growth_file):
    env_df = read_csv(env_file) if env_file is not None else pd.DataFrame()
    growth_df = read_csv(growth_file) if growth_file is not None else pd.DataFrame()
    env_df = std_cols(env_df)
    growth_df = std_cols(growth_df)
    return env_df, growth_df


with st.spinner("데이터를 읽는 중..."):
    env_df, growth_df = load_data(env_file, growth_file)

st.sidebar.markdown(f"**{uploaded_school}** 극지식물 분석 데이터")


# ──────────────────────────────────────────────
# 탭 구성
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔬 실험 개요", "🌡️ 환경 데이터", "🌱 생육 결과"])

# ──────────────────────────────────────────────
# 탭1: 실험 개요
# ──────────────────────────────────────────────
with tab1:
    try:
        st.header("🔬 실험 목적")
        st.markdown(
            """
**극지에서 자라는 식물은 혹독한 환경에서 살아남기 위해**
**땅속(지하부)으로 깊게 뿌리를 내리고,**
**그 힘이 땅 위(지상부) 생장에도 영향을 줍니다.**

이 대시보드는 우리 학교 극지식물 개체별 생육 결과를 분석하여
**지하부 생장이 지상부 생장에 실제로 영향을 미치는지**를 확인합니다.
"""
        )

        # 핵심 숫자 카드
        col_a, col_b, col_c, col_d = st.columns(4)

        if not growth_df.empty and "개체번호" in growth_df.columns:
            n = growth_df["개체번호"].nunique()
            col_a.metric("📋 분석 개체 수", f"{n}개체")
        else:
            col_a.metric("📋 분석 개체 수", "—")

        if not env_df.empty and "temperature" in env_df.columns:
            temp_series = pd.to_numeric(env_df["temperature"], errors="coerce").dropna()
            if len(temp_series) >= 2:
                avg_t = temp_series.mean()
                col_b.metric("🌡️ 평균 온도", f"{avg_t:.1f}°C")
            else:
                col_b.metric("🌡️ 평균 온도", "—")
        else:
            col_b.metric("🌡️ 평균 온도", "—")

        if not env_df.empty and "humidity" in env_df.columns:
            hum_series = pd.to_numeric(env_df["humidity"], errors="coerce").dropna()
            if len(hum_series) >= 2:
                avg_h = hum_series.mean()
                col_c.metric("💧 평균 습도", f"{avg_h:.1f}%")
            else:
                col_c.metric("💧 평균 습도", "—")
        else:
            col_c.metric("💧 평균 습도", "—")

        if not env_df.empty and "time" in env_df.columns:
            t_series = pd.to_datetime(env_df["time"], errors="coerce").dropna()
            if len(t_series) >= 2:
                span = f"{t_series.min().strftime('%m/%d')} ~ {t_series.max().strftime('%m/%d')}"
                col_d.metric("📅 측정 기간", span)
            else:
                col_d.metric("📅 측정 기간", "—")
        else:
            col_d.metric("📅 측정 기간", "—")

        st.divider()

        # 전하고 싶은 내용
        st.subheader("💡 전하고 싶은 내용")
        st.markdown(
            """
**지하부(뿌리) 생장이 지상부(줄기·잎) 생장에 영향을 미친다.**

동일한 환경에서 자란 극지식물이라도 지하부 길이와 질량이 클수록
지상부 길이·질량도 함께 커지는 경향이 관찰되었습니다.

즉, **땅속에서 영양과 수분을 잘 확보한 개체가 땅 위에서도 더 잘 자란다**는
사실을 데이터로 확인할 수 있습니다.
"""
        )

        # 생육 개체 요약 표
        if not growth_df.empty:
            numeric_cols = [
                c for c in ["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량", "생중량"]
                if c in growth_df.columns
            ]
            if numeric_cols:
                summary = growth_df[numeric_cols].describe().round(2)
                st.subheader("📊 개체별 생육 통계 요약")
                st.dataframe(summary, width="stretch")
            else:
                st.info("📊 생육 측정 항목이 없어 통계 요약을 표시할 수 없습니다.")
        else:
            st.info("생육 CSV를 업로드하면 개체별 통계 요약이 표시됩니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ──────────────────────────────────────────────
# 탭2: 환경 데이터
# ──────────────────────────────────────────────
with tab2:
    try:
        env_available = not env_df.empty and "time" in env_df.columns

        if not env_available:
            st.info("🌡️ 환경 CSV를 업로드하면 이 탭의 내용이 표시됩니다.")
        else:
            # ── 날짜별 온도·습도 꺾은선 그래프 ──
            env_plot_df = env_df.dropna(subset=["time"]).copy()
            env_plot_df["time"] = pd.to_datetime(env_plot_df["time"], errors="coerce")
            env_plot_df = env_plot_df.dropna(subset=["time"])

            num_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]

            if num_cols:
                # 하루 단위 평균으로 집계 (데이터가 많으면 그래프가 복잡해지므로)
                env_plot_df["date"] = env_plot_df["time"].dt.floor("D")

                daily = env_plot_df.groupby("date")[num_cols].mean().reset_index()

                fig_env = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.12,
                    subplot_titles=["날짜별 평균 온도 변화", "날짜별 평균 습도 변화"],
                )

                colors = {"temperature": "#e74c3c", "humidity": "#3498db", "ph": "#2ecc71", "ec": "#f39c12"}
                names = {"temperature": "온도(°C)", "humidity": "습도(%)", "ph": "pH", "ec": "EC(mS/cm)"}

                for i, col in enumerate(num_cols):
                    row = 1 if col in ("temperature", "ph", "ec") else 2
                    fig_env.add_trace(
                        go.Scatter(
                            x=daily["date"],
                            y=daily[col],
                            mode="lines+markers",
                            name=names.get(col, col),
                            line=dict(color=colors.get(col, "#888"), width=2),
                            marker=dict(size=4),
                            hovertemplate=f"{col}<br>%{{x|%m/%d %H:%M}}<br>%{{y:.2f}}<extra></extra>",
                        ),
                        row=row, col=1,
                    )

                fig_env.update_layout(
                    height=520,
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                fig_env.update_xaxes(title_text="날짜", row=2, col=1)
                fig_env.update_yaxes(title_text="°C", row=1, col=1)
                fig_env.update_yaxes(title_text="%", row=2, col=1)

                st.plotly_chart(fig_env, width="stretch")
            else:
                st.info("온도와 습도 열이 없어 환경 그래프를 표시할 수 없습니다.")

            # ── 환경 데이터 원본 표 ──
            st.subheader("📋 환경 데이터 원본")
            display_cols = ["time"] + num_cols
            display_cols = [c for c in display_cols if c in env_df.columns]
            if display_cols:
                env_display = env_df[display_cols].copy()
                if "time" in env_display.columns:
                    env_display["time"] = pd.to_datetime(env_display["time"], errors="coerce").dt.strftime(
                        "%Y-%m-%d %H:%M"
                    )
                env_display = env_display.dropna(how="all")
                st.dataframe(env_display, width="stretch", use_container_width=False)

                # 다운로드 버튼
                csv_bytes = env_display.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 환경 데이터 다운로드 (CSV)",
                    csv_bytes,
                    file_name="환경데이터_download.csv",
                    mime="text/csv",
                )
            else:
                st.info("표시할 환경 열이 없습니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ──────────────────────────────────────────────
# 탭3: 생육 결과
# ──────────────────────────────────────────────
with tab3:
    try:
        growth_available = not growth_df.empty and "개체번호" in growth_df.columns

        if not growth_available:
            st.info("🌱 생육 CSV를 업로드하면 이 탭의 내용이 표시됩니다.")
        else:
            # 사용할 열만 골라냄 (없는 열은 건너뛰기)
            gcols = [
                c for c in ["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량", "생중량"]
                if c in growth_df.columns
            ]

            if not gcols:
                st.info("생육 측정 항목이 없어 생육 그래프를 표시할 수 없습니다.")
            else:
                gf = growth_df[["개체번호"] + gcols].copy()
                # 숫자 열만 유지, 개체번호는 문자열로
                for c in gcols:
                    gf[c] = pd.to_numeric(gf[c], errors="coerce")
                gf = gf.dropna(subset=gcols, how="all")

                # ── 개체별 생장 막대 그래프 ──
                # 선택한 측정 항목들 각각을 막대 그래프로 (시리즈를 쌓아 표시하지 않고 개별 그래프)
                # 생중량 우선: 있으면 생중량 막대 + 상위/하위 강조
                if "생중량" in gcols and gf["생중량"].notna().sum() >= 2:
                    gf_sorted = gf.sort_values("생중량", ascending=False).reset_index(drop=True)
                    n_total = len(gf_sorted)
                    top_n = max(3, min(5, n_total))
                    bottom_n = max(3, min(5, n_total))

                    top_ids = gf_sorted["개체번호"].iloc[:top_n].tolist()
                    bottom_ids = gf_sorted["개체번호"].iloc[-bottom_n:].tolist()

                    # 색상: 상위=초록, 하위=주황, 중간=회색
                    colors_bar = []
                    for idx, row in gf_sorted.iterrows():
                        if row["개체번호"] in top_ids:
                            colors_bar.append("#27ae60")
                        elif row["개체번호"] in bottom_ids:
                            colors_bar.append("#e67e22")
                        else:
                            colors_bar.append("#95a5a6")

                    fig_bar = go.Figure()
                    fig_bar.add_trace(
                        go.Bar(
                            x=gf_sorted["개체번호"].tolist(),
                            y=gf_sorted["생중량"].tolist(),
                            marker_color=colors_bar,
                            hovertemplate="개체: %{{x}}<br>생중량: %{{y:.2f}}g<extra></extra>",
                        )
                    )

                    # 상위/하위 주석선
                    top_avg = gf_sorted.iloc[:top_n]["생중량"].mean()
                    bottom_avg = gf_sorted.iloc[-bottom_n:]["생중량"].mean()
                    fig_bar.add_hline(
                        y=top_avg, line_dash="dot", line_color="#27ae60",
                        annotation_text=f"상위 {top_n}개체 평균: {top_avg:.2f}g",
                        annotation_position="top right", annotation_font=dict(size=11),
                    )
                    fig_bar.add_hline(
                        y=bottom_avg, line_dash="dot", line_color="#e67e22",
                        annotation_text=f"하위 {bottom_n}개체 평균: {bottom_avg:.2f}g",
                        annotation_position="top right", annotation_font=dict(size=11),
                    )

                    fig_bar.update_layout(
                        title="📊 개체별 생중량(g) 비교 ( 진한 초록=상위 · 주황=하위 )",
                        xaxis_title="개체번호",
                        yaxis_title="생중량 (g)",
                        height=400,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        margin=dict(l=50, r=30, t=50, b=40),
                    )
                    st.plotly_chart(fig_bar, width="stretch")

                    # 상위/하위 표
                    col_top, col_bot = st.columns(2)
                    with col_top:
                        st.markdown(f"**🏆 생중량 상위 {top_n}개체**")
                        top_df = gf_sorted.iloc[:top_n][["개체번호", "생중량"] + [c for c in gcols if c != "생중량"]]
                        st.dataframe(top_df.round(2), width="stretch", use_container_width=False)
                    with col_bot:
                        st.markdown(f"**🔻 생중량 하위 {bottom_n}개체**")
                        bot_df = gf_sorted.iloc[-bottom_n:][["개체번호", "생중량"] + [c for c in gcols if c != "생중량"]]
                        st.dataframe(bot_df.round(2), width="stretch", use_container_width=False)

                # ── 선택된 생육 항목 막대 그래프 (생중량 제외) ──
                bar_cols = [c for c in gcols if c != "생중량"]
                if bar_cols:
                    fig_bar2 = make_subplots(
                        rows=len(bar_cols), cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.08,
                        subplot_titles=bar_cols,
                    )
                    for i, col in enumerate(bar_cols):
                        sub = gf.dropna(subset=[col]).sort_values(col, ascending=False)
                        fig_bar2.add_trace(
                            go.Bar(
                                x=sub["개체번호"].tolist(),
                                y=sub[col].tolist(),
                                marker_color="#3498db",
                                hovertemplate=f"{col}<br>개체: %{{x}}<br>값: %{{y:.2f}}<extra></extra>",
                            ),
                            row=i + 1, col=1,
                        )
                        fig_bar2.update_yaxes(title_text=col, row=i + 1, col=1)
                    fig_bar2.update_layout(
                        title="📊 개체별 생육 측정 항목 비교",
                        height=180 + 140 * len(bar_cols),
                        showlegend=False,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        margin=dict(l=50, r=20, t=40, b=30),
                    )
                    fig_bar2.update_xaxes(title_text="개체번호")
                    st.plotly_chart(fig_bar2, width="stretch")

                # ── 지하부 길이 vs 지상부 길이 산점도 ──
                if {"지하부 길이", "지상부 길이"} <= set(gcols):
                    df_scatter = gf.dropna(subset=["지하부 길이", "지상부 길이"])
                    if len(df_scatter) >= 2:
                        x = df_scatter["지하부 길이"].values
                        y = df_scatter["지상부 길이"].values

                        # 추세선 (np.polyfit)
                        slope, intercept = np.polyfit(x, y, 1)
                        x_line = np.linspace(x.min(), x.max(), 100)
                        y_line = slope * x_line + intercept

                        # 피어슨 상관계수
                        r, p_val = stats.pearsonr(x, y)

                        fig_scatter = go.Figure()
                        fig_scatter.add_trace(
                            go.Scatter(
                                x=x,
                                y=y,
                                mode="markers",
                                marker=dict(
                                    size=10,
                                    color="#2980b9",
                                    line=dict(width=1, color="white"),
                                    opacity=0.85,
                                ),
                                hovertemplate="지하부 길이: %{{x:.1f}}mm<br>지상부 길이: %{{y:.1f}}mm<extra></extra>",
                            )
                        )
                        fig_scatter.add_trace(
                            go.Scatter(
                                x=x_line,
                                y=y_line,
                                mode="lines",
                                line=dict(color="#e74c3c", width=2, dash="dot"),
                                name=f"추세선 (기울기={slope:.3f})",
                                hoverinfo="skip",
                            )
                        )

                        fig_scatter.update_layout(
                            title=(
                                f"🔗 지하부 길이 vs 지상부 길이  "
                                f"(상관계수 r = {r:.3f}"
                                f"{' · 유의함(p<0.05)' if p_val < 0.05 else ' · 유의하지 않음'}"
                                f")"
                            ),
                            xaxis_title="지하부 길이 (mm)",
                            yaxis_title="지상부 길이 (mm)",
                            height=460,
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            margin=dict(l=50, r=30, t=50, b=40),
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        )

                        st.plotly_chart(fig_scatter, width="stretch")

                    else:
                        st.info("지하부 길이와 지상부 길이 데이터가 부족해 산점도를 표시할 수 없습니다.")

                # ── 잎 수 vs 생중량 산점도 ──
                if {"잎 수", "생중량"} <= set(gcols):
                    df_scatter2 = gf.dropna(subset=["잎 수", "생중량"])
                    if len(df_scatter2) >= 2:
                        x2 = df_scatter2["잎 수"].values
                        y2 = df_scatter2["생중량"].values
                        slope2, intercept2 = np.polyfit(x2, y2, 1)
                        x_line2 = np.linspace(x2.min(), x2.max(), 100)
                        y_line2 = slope2 * x_line2 + intercept2
                        r2, p_val2 = stats.pearsonr(x2, y2)

                        fig_scatter2 = go.Figure()
                        fig_scatter2.add_trace(
                            go.Scatter(
                                x=x2,
                                y=y2,
                                mode="markers",
                                marker=dict(
                                    size=10,
                                    color="#27ae60",
                                    line=dict(width=1, color="white"),
                                    opacity=0.85,
                                ),
                                hovertemplate="잎 수: %{{x:.0f}}장<br>생중량: %{{y:.2f}}g<extra></extra>",
                            )
                        )
                        fig_scatter2.add_trace(
                            go.Scatter(
                                x=x_line2,
                                y=y_line2,
                                mode="lines",
                                line=dict(color="#e74c3c", width=2, dash="dot"),
                                name=f"추세선 (기울기={slope2:.3f})",
                                hoverinfo="skip",
                            )
                        )

                        fig_scatter2.update_layout(
                            title=(
                                f"🍃 잎 수 vs 생중량  "
                                f"(상관계수 r = {r2:.3f}"
                                f"{' · 유의함(p<0.05)' if p_val2 < 0.05 else ' · 유의하지 않음'}"
                                f")"
                            ),
                            xaxis_title="잎 수 (장)",
                            yaxis_title="생중량 (g)",
                            height=420,
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            margin=dict(l=50, r=30, t=50, b=40),
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        )
                        st.plotly_chart(fig_scatter2, width="stretch")

                # ── 생육 원본 데이터 표 + 다운로드 ──
                st.divider()
                st.subheader("📋 생육 결과 데이터 원본")
                display_gcols = ["개체번호"] + gcols
                gf_display = gf[display_gcols].copy()
                st.dataframe(gf_display.round(2), width="stretch", use_container_width=False)

                csv_g = gf_display.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 생육 데이터 다운로드 (CSV)",
                    csv_g,
                    file_name="생육결과_download.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
