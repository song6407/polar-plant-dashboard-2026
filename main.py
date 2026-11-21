import io
import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="송천고 극지식물 생장 데이터", layout="wide")

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

# 화면 상단에 제목 표시
st.title("송천고 극지 식물 데이터")


# ---------------------------------------------------------------------------
# 열 이름 표준화 (KeyError 방지)
# ---------------------------------------------------------------------------
def std_cols(df):
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
            num = pd.to_numeric(
                df[c].astype(str)
                .str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(
        df.columns
    ):
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


def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def school_name_from_filename(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    name = re.sub(r"[\s_\-.]+", " ", base)
    tokens = name.split()
    if tokens:
        return tokens[0]
    return "우리 학교"


def apply_kr_font(fig):
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )
    return fig


# ---------------------------------------------------------------------------
# 사이드바: 파일 업로드
# ---------------------------------------------------------------------------
st.sidebar.title("📁 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv")

env_df = pd.DataFrame()
growth_df = pd.DataFrame()
env_ok = False
growth_ok = False

if env_file is not None:
    with st.spinner("환경 데이터를 읽는 중..."):
        try:
            raw = read_csv(env_file)
            env_df = std_cols(raw)
            env_ok = not env_df.empty
        except Exception as e:
            st.warning(f"⚠️ 환경 CSV를 읽지 못했습니다: {e}")

if growth_file is not None:
    with st.spinner("생육 결과 데이터를 읽는 중..."):
        try:
            raw = read_csv(growth_file)
            growth_df = std_cols(raw)
            growth_ok = not growth_df.empty
        except Exception as e:
            st.warning(f"⚠️ 생육 CSV를 읽지 못했습니다: {e}")

env_school = school_name_from_filename(env_file.name) if env_file else ""
growth_school = school_name_from_filename(growth_file.name) if growth_file else ""
school_display = env_school or growth_school or "우리 학교"

st.sidebar.divider()
st.sidebar.caption(f"🏫 {school_display}")

st.sidebar.divider()
st.sidebar.subheader("📊 생육 측정 항목")
metric_cols = []
if growth_ok:
    candidates = [
        "잎 수",
        "지상부 길이",
        "지하부 길이",
        "생중량",
        "지상부 질량",
        "지하부 질량",
    ]
    metric_cols = [c for c in candidates if c in growth_df.columns]
selected_metric = st.sidebar.selectbox(
    "비교할 측정 항목",
    metric_cols,
    index=0 if metric_cols else None,
    disabled=not metric_cols,
)

st.sidebar.divider()
st.sidebar.info(
    "💡 아직 파일이 없습니다. 사이드바에서 CSV를 업로드하면 대시보드가 활성화됩니다."
)


# ---------------------------------------------------------------------------
# 탭 정의
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔬 실험 개요", "🌡️ 환경 변화", "🌱 개체별 생장"])


# ===========================================================================
# 탭 1 ─ 실험 목적 · 핵심 숫자 · 전달 메시지
# ===========================================================================
with tab1:
    try:
        st.header("🔬 실험 목적")
        st.markdown(
            """
- 같은 조건에서 자란 극지식물 개체들이 **얼마나 다르게 자라는지** 확인한다.
- 특히 **지하부(뿌리) 생장이 지상부 줄기 생장에 어떤 영향을 미치는지** 알아본다.
- 잎 수와 지상부 길이 중 **어느 것이 전체 무게(생중량)와 더 관련이 깊은지** 비교한다.
"""
        )
        st.divider()

        if env_ok and growth_ok:
            st.subheader(f"🏫 {school_display} 극지식물 생육 분석")

            # 환경 지표
            if "temperature" in env_df.columns:
                env_valid = env_df.dropna(subset=["temperature"])
                if len(env_valid) >= 2:
                    st.metric("평균 온도", f"{env_valid['temperature'].mean():.1f} ℃")
                    st.metric("최고 온도", f"{env_valid['temperature'].max():.1f} ℃")
                    st.metric("최저 온도", f"{env_valid['temperature'].min():.1f} ℃")
                    st.caption(f"📅 환경 측정 횟수: {len(env_valid)}회")
                else:
                    st.info("💡 환경 데이터 온도 값이 부족해 지표를 계산하지 못해요.")
            else:
                st.info("💡 환경 데이터의 온도 열이 없어 온도 지표를 표시할 수 없어요.")

            st.divider()

            # 생육 지표
            st.metric("개체 수", f"{len(growth_df)}개체")
            if "잎 수" in growth_df.columns:
                vals = growth_df["잎 수"].dropna()
                if len(vals) >= 1:
                    st.metric("평균 잎 수", f"{vals.mean():.1f}장")
            if "지상부 길이" in growth_df.columns:
                vals = growth_df["지상부 길이"].dropna()
                if len(vals) >= 1:
                    st.metric("평균 지상부 길이", f"{vals.mean():.1f} mm")
            if "지하부 길이" in growth_df.columns:
                vals = growth_df["지하부 길이"].dropna()
                if len(vals) >= 1:
                    st.metric("평균 지하부 길이", f"{vals.mean():.1f} mm")
            if "생중량" in growth_df.columns:
                vals = growth_df["생중량"].dropna()
                if len(vals) >= 1:
                    st.metric("평균 생중량", f"{vals.mean():.2f} g")

            st.divider()
            st.subheader("💡 전하고 싶은 내용")
            st.markdown(
                """
**지하부(뿌리) 생장이 지상부 생장에 영향을 미친다.**  
뿌리가 길고 튼튼하게 자란 개체일수록 줄기도 더 길게 자라는 경향이 있다.
"""
            )

            # 지하부 ↔ 지상부 상관계수
            if "지하부 길이" in growth_df.columns and "지상부 길이" in growth_df.columns:
                d = growth_df[["지하부 길이", "지상부 길이"]].dropna()
                if len(d) >= 2:
                    corr = d["지하부 길이"].corr(d["지상부 길이"])
                    note = ""
                    if corr > 0.5:
                        note = " (강한 양의 상관)"
                    elif abs(corr) < 0.3:
                        note = " (약한 상관)"
                    st.info(
                        f"📊 지하부 길이 ↔ 지상부 길이 상관계수: **{corr:.2f}**{note}"
                    )
                else:
                    st.info("💡 비교할 데이터가 충분하지 않아 상관계수를 계산하지 못했어요.")
        elif not env_ok and not growth_ok:
            st.info("💡 아직 업로드된 파일이 없습니다. 사이드바에서 CSV를 올려주세요.")
        elif not env_ok:
            st.info("💡 환경 CSV가 업로드되지 않아 환경 관련 수치를 표시할 수 없어요.")
        elif not growth_ok:
            st.info("💡 생육 CSV가 업로드되지 않아 생육 관련 수치를 표시할 수 없어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ===========================================================================
# 탭 2 ─ 날짜별 온도·습도 꺾은선 그래프 + 환경 원본 표
# ===========================================================================
with tab2:
    try:
        if not env_ok:
            st.info("💡 환경 CSV를 먼저 업로드해주세요.")
        else:
            st.header("🌡️ 날짜별 온도·습도 변화")
            env_t = env_df.dropna(subset=["time"]).sort_values("time").copy()

            if env_t.empty:
                st.info("💡 환경 데이터의 시간 열이 없거나 모두 비어 있어요.")
            else:
                fig_lines = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.12,
                    subplot_titles=("온도 변화", "습도 변화"),
                )

                has_temp = "temperature" in env_t.columns
                has_hum  = "humidity" in env_t.columns

                if has_temp:
                    fig_lines.add_trace(
                        go.Scatter(
                            x=env_t["time"],
                            y=env_t["temperature"],
                            mode="lines",
                            name="온도",
                            line=dict(color="#e74c3c", width=1.5),
                            hovertemplate="시간: %{x|%Y-%m-%d %H:%M}<br>온도: %{y:.1f} ℃<extra></extra>",
                        ),
                        row=1,
                        col=1,
                    )
                if has_hum:
                    fig_lines.add_trace(
                        go.Scatter(
                            x=env_t["time"],
                            y=env_t["humidity"],
                            mode="lines",
                            name="습도",
                            line=dict(color="#3498db", width=1.5),
                            hovertemplate="시간: %{x|%Y-%m-%d %H:%M}<br>습도: %{y:.1f} %<extra></extra>",
                        ),
                        row=2,
                        col=1,
                    )

                if has_temp:
                    fig_lines.update_yaxes(title_text="온도 (℃)", row=1, col=1)
                if has_hum:
                    fig_lines.update_yaxes(title_text="습도 (%)", row=2, col=1)
                fig_lines.update_xaxes(title_text="시간")
                fig_lines.update_layout(
                    height=420,
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                apply_kr_font(fig_lines)
                st.plotly_chart(fig_lines, width="stretch")

            st.divider()
            st.subheader("📄 환경 데이터 원본")
            display_cols = [
                c for c in ["time", "temperature", "humidity", "ph", "ec"]
                if c in env_df.columns
            ]
            if display_cols:
                st.dataframe(env_df[display_cols], width="stretch")
            else:
                st.info("💡 환경 데이터에서 표준 열 이름을 찾지 못했어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ===========================================================================
# 탭 3 ─ 개체별 생장 막대 그래프 + 지하부/지상부 산점도
# ===========================================================================
with tab3:
    try:
        if not growth_ok:
            st.info("💡 생육 CSV를 먼저 업로드해주세요.")
        else:
            g = growth_df.dropna(subset=["개체번호"]).copy()
            if g.empty:
                st.info("💡 생육 데이터의 개체번호 열이 없거나 모두 비어 있어요.")
            else:
                # 막대 그래프 헬퍼 함수
                def make_bar_graph(df, col, title, bar_color):
                    d = df[["개체번호", col]].dropna()
                    if d.empty or len(d) < 2:
                        return None
                    avg = d[col].mean()
                    d = d.sort_values(col, ascending=False)
                    colors = []
                    for val in d[col]:
                        if val >= avg * 1.15:
                            colors.append("#2ecc71")
                        elif val <= avg * 0.85:
                            colors.append("#e74c3c")
                        else:
                            colors.append(bar_color)
                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=d["개체번호"].astype(str),
                            y=d[col],
                            marker_color=colors,
                            hovertemplate="%{x}<br>%{y:.1f}<extra></extra>",
                        )
                    )
                    fig.add_hline(
                        y=avg,
                        line_dash="dot",
                        line_color="gray",
                        annotation_text=f"평균 {avg:.1f}",
                        annotation_position="top right",
                    )
                    fig.update_layout(
                        title=title,
                        xaxis_title="개체번호",
                        yaxis_title=col,
                        margin=dict(l=40, r=20, t=50, b=40),
                        height=340,
                    )
                    apply_kr_font(fig)
                    return fig

                st.header("🌱 개체별 생장 비교")

                # 선택된 측정항목으로 막대 그래프
                metric_col = None
                if selected_metric and selected_metric in g.columns:
                    metric_col = selected_metric
                if metric_col:
                    fig_bar = make_bar_graph(
                        g, metric_col,
                        f"개체별 {metric_col} 막대 그래프",
                        "#9b59b6",
                    )
                    if fig_bar:
                        st.plotly_chart(fig_bar, width="stretch")
                        avg_val = g[metric_col].dropna().mean()
                        top_ids = g.nlargest(5, metric_col)["개체번호"].tolist()
                        bot_ids = g.nsmallest(5, metric_col)["개체번호"].tolist()
                        st.caption(
                            "🟢 평균 이상 개체: "
                            + ", ".join(str(x) for x in top_ids)
                            + "    🔴 평균 이하 개체: "
                            + ", ".join(str(x) for x in bot_ids)
                        )
                    else:
                        st.info(f"💡 '{metric_col}' 데이터가 부족해 막대 그래프를 그리지 못해요.")
                else:
                    st.info("💡 선택할 측정 항목이 없어요. 생육 CSV 열 이름을 확인해주세요.")

                st.divider()
                st.subheader("🔗 지하부 길이 vs 지상부 길이 산점도")

                if "지하부 길이" in g.columns and "지상부 길이" in g.columns:
                    d_scatter = g[["지하부 길이", "지상부 길이"]].dropna()
                    if len(d_scatter) >= 2:
                        x = d_scatter["지하부 길이"].values
                        y = d_scatter["지상부 길이"].values
                        slope, intercept = np.polyfit(x, y, 1)
                        x_line = np.linspace(x.min(), x.max(), 100)
                        y_line = slope * x_line + intercept
                        r = np.corrcoef(x, y)[0, 1]

                        fig_scatter = go.Figure()
                        fig_scatter.add_trace(
                            go.Scatter(
                                x=x,
                                y=y,
                                mode="markers",
                                marker=dict(
                                    size=9,
                                    color="#2980b9",
                                    line=dict(width=1, color="white"),
                                ),
                                hovertemplate="지하부: %{x:.1f} mm<br>지상부: %{y:.1f} mm<extra></extra>",
                                name="개체",
                            )
                        )
                        fig_scatter.add_trace(
                            go.Scatter(
                                x=x_line,
                                y=y_line,
                                mode="lines",
                                line=dict(color="#e74c3c", width=2, dash="dash"),
                                name=f"추세선 (r={r:.2f})",
                                hoverinfo="skip",
                            )
                        )
                        fig_scatter.update_layout(
                            title=f"지하부 길이 vs 지상부 길이 (상관계수 r = {r:.2f})",
                            xaxis_title="지하부 길이 (mm)",
                            yaxis_title="지상부 길이 (mm)",
                            width=640,
                            height=440,
                            margin=dict(l=50, r=20, t=50, b=40),
                        )
                        apply_kr_font(fig_scatter)
                        st.plotly_chart(fig_scatter, width="stretch")

                        if r > 0.5:
                            st.success(
                                f"💡 지하부 길이와 지상부 길이 사이에 **뚜렷한 양의 상관**이 있어요 (r = {r:.2f}). "
                                "뿌리가 잘 자란 개체가 줄기도 길게 자라는 경향이 확인됩니다."
                            )
                        elif r > 0.3:
                            st.info(
                                f"💡 지하부 길이와 지상부 길이 사이에 **약한 양의 상관**이 있어요 (r = {r:.2f})."
                            )
                        else:
                            st.info(
                                f"💡 지하부 길이와 지상부 길이의 상관이 약해요 (r = {r:.2f}). "
                                "다른 요인이 생장에 더 큰 영향을 줄 수 있어요."
                            )
                    else:
                        st.info("💡 지하부·지상부 길이 데이터가 2개 미만이라 산점도를 그리지 못해요.")
                else:
                    st.info("💡 '지하부 길이'와 '지상부 길이' 열이 모두 있어야 산점도를 그릴 수 있어요.")

                # 무거운 개체 vs 가벼운 개체 비교
                st.divider()
                st.subheader("⚖️ 무거운 개체 vs 가벼운 개체 비교")

                if "생중량" in g.columns:
                    d_w = g[["개체번호", "생중량"]].dropna()
                    if len(d_w) >= 4:
                        mid = d_w["생중량"].median()
                        heavy = g[g["생중량"] >= mid].dropna(subset=["생중량"])
                        light = g[g["생중량"] < mid].dropna(subset=["생중량"])
                        cmp_cols = [
                            c for c in ["잎 수", "지상부 길이", "지하부 길이"]
                            if c in g.columns
                        ]
                        if cmp_cols:
                            rows = []
                            rows.append(["구분", "개체 수"] + cmp_cols)
                            rows.append(
                                ["무거운 개체 (중앙값 이상)", len(heavy)]
                                + [
                                    f"{heavy[c].mean():.1f}"
                                    if c in heavy.columns and heavy[c].notna().any()
                                    else "N/A"
                                    for c in cmp_cols
                                ]
                            )
                            rows.append(
                                ["가벼운 개체 (중앙값 미만)", len(light)]
                                + [
                                    f"{light[c].mean():.1f}"
                                    if c in light.columns and light[c].notna().any()
                                    else "N/A"
                                    for c in cmp_cols
                                ]
                            )
                            cmp_df = pd.DataFrame(rows[1:], columns=rows[0])
                            st.dataframe(cmp_df, width="stretch", hide_index=True)
                        else:
                            st.info("💡 비교할 생육 측정 항목(잎 수, 길이 등)이 없어요.")
                    else:
                        st.info("💡 개체 수가 너무 적어 무거운/가벼운 개체 비교가 어려워요.")
                else:
                    st.info("💡 생육 데이터에 '생중량' 열이 없어요. (지상부 질량 + 지하부 질량으로 자동 계산됩니다.)")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ---------------------------------------------------------------------------
# 다운로드 버튼 (한글 깨짐 방지: utf-8-sig)
# ---------------------------------------------------------------------------
if env_ok and "time" in env_df.columns:
    csv_env = env_df.to_csv(index=False).encode("utf-8-sig")
    st.sidebar.download_button(
        "📥 환경 CSV 다운로드",
        csv_env,
        "환경데이터_정규화.csv",
        "text/csv",
    )

if growth_ok and "개체번호" in growth_df.columns:
    csv_growth = growth_df.to_csv(index=False).encode("utf-8-sig")
    st.sidebar.download_button(
        "📥 생육 CSV 다운로드",
        csv_growth,
        "생육결과_정규화.csv",
        "text/csv",
    )
