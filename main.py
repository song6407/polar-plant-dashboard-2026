# code_execution: main.py 전체 코드 저장
code = r'''import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── 한글 폰트 CSS ───
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

st.set_page_config(page_title="송천고 극지식물 생장 데이터", layout="wide")
st.title("🌿 송천고 극지식물 생장 데이터")

# ─── 인코딩 안전 CSV 읽기 ───
def read_csv_safe(data_bytes):
    """utf-8-sig → cp949 순서로 시도"""
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data_bytes), encoding=enc)
        except Exception:
            continue
    return pd.DataFrame()

# ─── 열 이름 표준화 ───
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
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
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
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + \
                          pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df


@st.cache_data
def load_env_data(file_bytes):
    df = read_csv_safe(file_bytes)
    if df.empty:
        return df
    return std_cols(df)


@st.cache_data
def load_growth_data(file_bytes):
    df = read_csv_safe(file_bytes)
    if df.empty:
        return df
    return std_cols(df)


# ─── 사이드바: 파일 업로드 ───
st.sidebar.header("📁 CSV 업로드")

env_file = st.sidebar.file_uploader("환경 데이터 CSV", type=["csv"])
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type=["csv"])

env_df = pd.DataFrame()
growth_df = pd.DataFrame()

if env_file is not None:
    with st.spinner("환경 데이터를 불러오는 중..."):
        env_df = load_env_data(env_file.getvalue())
        school_name = env_file.name.split("_")[0] if "_" in env_file.name else env_file.name
        st.sidebar.success(f"✅ 환경 데이터 로드 완료 — {school_name}")

if growth_file is not None:
    with st.spinner("생육 데이터를 불러오는 중..."):
        growth_df = load_growth_data(growth_file.getvalue())
        school_name2 = growth_file.name.split("_")[0] if "_" in growth_file.name else growth_file.name
        st.sidebar.success(f"✅ 생육 데이터 로드 완료 — {school_name2}")

# ─── 생육 측정 항목 선택 ───
available_growth = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량",
                                 "지상부 질량", "지하부 질량"] if c in growth_df.columns]
selected_metric = st.sidebar.selectbox(
    "📊 생육 측정 항목 선택",
    options=available_growth if available_growth else ["(데이터 없음)"],
    index=0 if available_growth else 0,
)

if env_df.empty and growth_df.empty:
    st.info("📎 사이드바에서 환경 데이터 CSV와 생육 결과 데이터 CSV를 업로드해주세요.")
    st.stop()

# 학교명 표시
sn = ""
if env_file is not None:
    sn = env_file.name
elif growth_file is not None:
    sn = growth_file.name
if sn:
    display_name = sn.split("_")[0] if "_" in sn else sn
    st.caption(f"학교: {display_name} · 파일 기준")


# ═══════════════════════════════════════════════════════════════
# 탭 1: 실험 목적 & 핵심 지표
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📌 실험 목적 & 핵심 지표", "📊 환경 데이터 분석", "🔬 개체별 생장 분석"])

with tab1:
    try:
        st.header("🔬 실험 목적")
        st.markdown("""
**이 실험의 핵심 질문**
1. **지하부(뿌리) 생장이 지상부(줄기·잎) 생장에 영향을 미치는가?**
2. **잎 수와 지상부 길이 중 어느 것이 생중량(전체 무게)과 더 밀접할까?**
3. **무거운 개체와 가벼운 개체는 어떤 생육 특징에서 차이가 나는가?**
        """.strip())

        # 핵심 숫자 카드
        n_individuals = len(growth_df["개체번호"].dropna()) if not growth_df.empty else 0
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="🧬 분석 개체 수",
                value=n_individuals,
                delta="개체",
                help="생육 결과 데이터에 등록된 총 개체 수",
            )

        with col2:
            if "temperature" in env_df.columns:
                avg_temp = env_df["temperature"].dropna().mean()
                st.metric(
                    label="🌡️ 평균 온도",
                    value=f"{avg_temp:.1f} °C",
                    delta=f"범위: {env_df['temperature'].dropna().min():.1f} ~ {env_df['temperature'].dropna().max():.1f} °C",
                    help="환경 데이터 전체 기간의 평균 온도",
                )
            else:
                st.metric(label="🌡️ 평균 온도", value="데이터 없음")

        with col3:
            if "humidity" in env_df.columns:
                avg_hum = env_df["humidity"].dropna().mean()
                st.metric(
                    label="💧 평균 습도",
                    value=f"{avg_hum:.1f} %",
                    delta=f"범위: {env_df['humidity'].dropna().min():.1f} ~ {env_df['humidity'].dropna().max():.1f} %",
                    help="환경 데이터 전체 기간의 평균 습도",
                )
            else:
                st.metric(label="💧 평균 습도", value="데이터 없음")

        st.divider()

        # 생육 지표 요약 카드
        st.subheader("📊 생육 지표 요약")
        growth_cols_avail = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                             if c in growth_df.columns]
        if growth_cols_avail:
            metrics_data = []
            for m in growth_cols_avail:
                s = growth_df[m].dropna()
                if len(s) >= 2:
                    metrics_data.append({
                        "지표": m,
                        "평균": round(s.mean(), 2),
                        "최대": round(s.max(), 2),
                        "최소": round(s.min(), 2),
                        "표준편차": round(s.std(), 2),
                        "n": len(s),
                    })
            if metrics_data:
                df_metrics = pd.DataFrame(metrics_data)
                st.dataframe(df_metrics, use_container_width=True, hide_index=True)
                st.caption("💡 표준편차: 값이 평균에서 얼마나 퍼져 있는지를 나타냅니다. 값이 클수록 개체 간 차이가 큽니다.")
        else:
            st.info("📎 생육 데이터 CSV를 업로드하면 생육 지표 요약표가 표시됩니다.")

        st.divider()

        # 핵심 메시지
        st.subheader("💡 전하고 싶은 내용")
        st.markdown("""
**지하부 생장이 지상부 생장에 영향을 미친다.**

아래 탭3의 산점도(지하부 길이 vs 지상부 길이)와 상관계수를 통해
뿌리(지하부)가 잘 자란 개체일수록 줄기·잎(지상부)도 크다는 경향이 확인됩니다.
        """.strip())

        # 상관계수 미리 보기
        if "지하부 길이" in growth_df.columns and "지상부 길이" in growth_df.columns:
            sub = growth_df[["지하부 길이", "지상부 길이"]].dropna()
            if len(sub) >= 2:
                r = sub["지하부 길이"].corr(sub["지상부 길이"])
                slope, intercept = np.polyfit(sub["지하부 길이"], sub["지상부 길이"], 1)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.info(f"**상관계수 r = {r:.3f}**  (n={len(sub)})")
                    st.caption("1에 가까울수록 강한 양의 상관관계" if r > 0 else "음의 상관관계")
                with col_b:
                    st.info(f"**추세선**: y(지상부) = {slope:.3f} × x(지하부) + {intercept:.2f}")
                    st.caption(f"지하부 1mm ↑ → 지상부 약 {slope:.2f}mm ↑")
        else:
            st.info("📎 생육 데이터에 '지하부 길이'와 '지상부 길이' 열이 있어야 상관계수를 계산할 수 있습니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ═══════════════════════════════════════════════════════════════
# 탭 2: 환경 데이터 분석
# ═══════════════════════════════════════════════════════════════
with tab2:
    try:
        if env_df.empty:
            st.info("📎 사이드바에서 환경 데이터 CSV를 업로드해주세요.")
        else:
            # 날짜별 온도 꺾은선
            if "time" in env_df.columns and "temperature" in env_df.columns:
                env_t = env_df[["time", "temperature"]].dropna().sort_values("time")
                if len(env_t) >= 2:
                    daily_temp = env_t.set_index("time").resample("D")["temperature"].mean().dropna()
                    fig_temp = go.Figure()
                    fig_temp.add_trace(go.Scatter(
                        x=daily_temp.index,
                        y=daily_temp.values,
                        mode="lines+markers",
                        name="일별 평균 온도",
                        line=dict(color="#f97316", width=2),
                        marker=dict(size=5, color="#f97316"),
                    ))
                    fig_temp.add_hline(
                        y=env_t["temperature"].mean(),
                        line_dash="dash",
                        line_color="#dc2626",
                        annotation_text=f"평균 {env_t['temperature'].mean():.1f}°C",
                        annotation_position="top right",
                    )
                    fig_temp.update_layout(
                        title="🌡️ 날짜별 평균 온도 변화",
                        xaxis_title="날짜",
                        yaxis_title="온도 (°C)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        height=380,
                        margin=dict(l=40, r=20, t=50, b=40),
                    )
                    fig_temp.update_yaxes(range=[0, 40])
                    st.plotly_chart(fig_temp, width="stretch")
                else:
                    st.info("📎 온도 데이터가 충분하지 않아 그래프를 표시할 수 없습니다.")
            else:
                st.info("📎 환경 데이터에 'time'과 'temperature' 열이 필요합니다.")

            # 날짜별 습도 꺾은선
            if "time" in env_df.columns and "humidity" in env_df.columns:
                env_h = env_df[["time", "humidity"]].dropna().sort_values("time")
                if len(env_h) >= 2:
                    daily_hum = env_h.set_index("time").resample("D")["humidity"].mean().dropna()
                    fig_hum = go.Figure()
                    fig_hum.add_trace(go.Scatter(
                        x=daily_hum.index,
                        y=daily_hum.values,
                        mode="lines+markers",
                        name="일별 평균 습도",
                        line=dict(color="#3b82f6", width=2),
                        marker=dict(size=5, color="#3b82f6"),
                    ))
                    fig_hum.add_hline(
                        y=env_h["humidity"].mean(),
                        line_dash="dash",
                        line_color="#1d4ed8",
                        annotation_text=f"평균 {env_h['humidity'].mean():.1f}%",
                        annotation_position="top right",
                    )
                    fig_hum.update_layout(
                        title="💧 날짜별 평균 습도 변화",
                        xaxis_title="날짜",
                        yaxis_title="습도 (%)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        height=380,
                        margin=dict(l=40, r=20, t=50, b=40),
                    )
                    fig_hum.update_yaxes(range=[0, 100])
                    st.plotly_chart(fig_hum, width="stretch")
                else:
                    st.info("📎 습도 데이터가 충분하지 않아 그래프를 표시할 수 없습니다.")
            else:
                st.info("📎 환경 데이터에 'humidity' 열이 없습니다.")

            # 환경 데이터 원본 표
            st.subheader("📋 환경 데이터 원본 (일부)")
            display_cols = [c for c in ["time", "temperature", "humidity", "ph", "ec"]
                           if c in env_df.columns]
            if display_cols:
                df_show = env_df[display_cols].copy()
                df_show["time"] = df_show["time"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(df_show, use_container_width=True, hide_index=True)
                st.caption(f"총 {len(df_show)}행 중 앞부분 표시")
            else:
                st.info("📎 표시할 환경 데이터 열이 없습니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ═══════════════════════════════════════════════════════════════
# 탭 3: 개체별 생장 분석
# ═══════════════════════════════════════════════════════════════
with tab3:
    try:
        if growth_df.empty:
            st.info("📎 사이드바에서 생육 결과 데이터 CSV를 업로드해주세요.")
        else:
            # 개체별 생중량 막대 그래프 (상위·하위 강조)
            if "생중량" in growth_df.columns:
                g_sorted = growth_df[["개체번호", "생중량"]].dropna().sort_values(
                    "생중량", ascending=True
                ).reset_index(drop=True)
                n = len(g_sorted)
                if n >= 2:
                    top_n = max(1, n // 4)
                    low_n = max(1, n // 4)
                    bar_colors = []
                    for i in range(n):
                        if i < low_n:
                            bar_colors.append("#f59e0b")  # 하위 25% — 주황
                        elif i >= n - top_n:
                            bar_colors.append("#10b981")  # 상위 25% — 초록
                        else:
                            bar_colors.append("#94a3b8")  # 중간 — 회색

                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=g_sorted["개체번호"].astype(str),
                        y=g_sorted["생중량"],
                        marker_color=bar_colors,
                        marker_line_color="white",
                        marker_line_width=0.5,
                        name="생중량",
                    ))
                    fig_bar.add_hline(
                        y=g_sorted["생중량"].mean(),
                        line_dash="dash",
                        line_color="#6366f1",
                        line_width=2,
                        annotation_text=f"평균 {g_sorted['생중량'].mean():.2f}g",
                        annotation_position="top right",
                    )
                    fig_bar.update_layout(
                        title=f"🌿 개체별 생중량 (총 {n}개체)  ·  초록=상위 25%  주황=하위 25%",
                        xaxis_title="개체번호",
                        yaxis_title="생중량 (g)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        height=420,
                        margin=dict(l=40, r=20, t=55, b=40),
                    )
                    fig_bar.update_xaxes(tickangle=90, tickfont=dict(size=8))
                    st.plotly_chart(fig_bar, width="stretch")
                else:
                    st.info("📎 생중량 데이터가 2개 미만이라 그래프를 표시할 수 없습니다.")

            # 지하부 길이 vs 지상부 길이 산점도 + 추세선
            if "지하부 길이" in growth_df.columns and "지상부 길이" in growth_df.columns:
                sub = growth_df[["개체번호", "지하부 길이", "지상부 길이", "생중량"]].dropna()
                if len(sub) >= 2:
                    x = sub["지하부 길이"].values
                    y = sub["지상부 길이"].values
                    r = np.corrcoef(x, y)[0, 1]
                    slope, intercept = np.polyfit(x, y, 1)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = slope * x_line + intercept

                    fig_scatter = make_subplots(rows=1, cols=1)

                    fig_scatter.add_trace(go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        name="개체",
                        marker=dict(
                            size=9,
                            color="#2d6a9f",
                            opacity=0.65,
                            line=dict(color="white", width=0.8),
                        ),
                    ))

                    fig_scatter.add_trace(go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name=f"추세선 (r={r:.3f})",
                        line=dict(color="#ef4444", width=2.5, dash="dash"),
                    ))

                    # 상위 3개체 강조
                    top3 = sub.nlargest(3, "생중량")
                    fig_scatter.add_trace(go.Scatter(
                        x=top3["지하부 길이"],
                        y=top3["지상부 길이"],
                        mode="markers",
                        name="생중량 상위 3",
                        marker=dict(size=14, color="#10b981", line=dict(color="#047857", width=2)),
                    ))

                    # 하위 3개체 강조
                    bot3 = sub.nsmallest(3, "생중량")
                    fig_scatter.add_trace(go.Scatter(
                        x=bot3["지하부 길이"],
                        y=bot3["지상부 길이"],
                        mode="markers",
                        name="생중량 하위 3",
                        marker=dict(size=14, color="#f97316", line=dict(color="#c2410c", width=2)),
                    ))

                    fig_scatter.update_layout(
                        title=f"🔬 지하부 길이 vs 지상부 길이  (상관계수 r = {r:.3f}, n={len(sub)})",
                        xaxis_title="지하부 길이 (mm)",
                        yaxis_title="지상부 길이 (mm)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        height=480,
                        margin=dict(l=50, r=30, t=55, b=40),
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    )
                    fig_scatter.update_xaxes(range=[0, x.max() * 1.1])
                    fig_scatter.update_yaxes(range=[0, y.max() * 1.1])

                    # 통계 주석
                    stats_text = (
                        f"n = {len(sub)}개체<br>"
                        f"r = {r:.3f}<br>"
                        f"기울기 = {slope:.3f}<br>"
                        f"절편 = {intercept:.2f}<br>"
                        f"→ 지하부 1mm ↑ 시 지상부 약 {slope:.2f}mm ↑"
                    )
                    fig_scatter.add_annotation(
                        text=stats_text,
                        x=0.97,
                        y=0.03,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        bgcolor="#eef2ff",
                        bordercolor="#6366f1",
                        borderwidth=1,
                        borderpad=8,
                        font=dict(size=11, family="monospace"),
                    )
                    st.plotly_chart(fig_scatter, width="stretch")

                    # 상관계수 해석
                    if abs(r) > 0.5:
                        st.success(f"✅ r = {r:.3f} — **뚜렷한 양의 상관관계** (지하부가 클수록 지상부도 큼 → 가설 지지)")
                    elif abs(r) > 0.3:
                        st.info(f"ℹ️ r = {r:.3f} — **중간 정도의 상관관계**")
                    else:
                        st.warning(f"⚠️ r = {r:.3f} — **상관관계가 약함** (다른 요인이 더 클 수 있음)")
                else:
                    st.info("📎 지하부/지상부 길이 데이터가 2개 미만입니다.")
            else:
                st.info("📎 생육 데이터에 '지하부 길이'와 '지상부 길이' 열이 필요합니다.")

            # 개체별 생육 지표 분포 (박스플롯)
            box_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이"]
                        if c in growth_df.columns]
            if box_cols:
                fig_box = make_subplots(
                    rows=1, cols=len(box_cols),
                    subplot_titles=box_cols,
                    horizontal_spacing=0.12,
                )
                colors = ["#8b5cf6", "#3b82f6", "#10b981"]
                for i, col in enumerate(box_cols):
                    s = growth_df[col].dropna()
                    fig_box.add_trace(go.Box(
                        y=s.values,
                        name=col,
                        boxpoints="outliers",
                        marker_color=colors[i % len(colors)],
                        marker_line_color="white",
                        line_color=colors[i % len(colors)],
                        fillcolor=colors[i % len(colors)] + "40",
                    ), row=1, col=i + 1)
                    mean_val = s.mean()
                    fig_box.add_hline(
                        y=mean_val,
                        line_dash="dot",
                        line_color="#1e293b",
                        line_width=1.5,
                        annotation_text=f"평균 {mean_val:.1f}",
                        annotation_position="top right",
                        row=1, col=i + 1,
                    )
                    fig_box.update_xaxes(showticklabels=False, row=1, col=i + 1)
                    fig_box.update_yaxes(title_text=col, row=1, col=i + 1,
                                         title_font=dict(size=11))
                fig_box.update_layout(
                    title=f"📊 {len(growth_df)}개체 생육 지표 분포",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    height=320,
                    margin=dict(l=20, r=20, t=45, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_box, width="stretch")

            # 하위 섹션: 무거운 개체 vs 가벼운 개체 비교
            if "생중량" in growth_df.columns:
                st.subheader("⚖️ 무거운 개체 vs 가벼운 개체 비교")
                g = growth_df[["개체번호", "생중량"] + box_cols].dropna()
                if len(g) >= 6:
                    g_sorted = g.sort_values("생중량")
                    n = len(g_sorted)
                    top_n = max(1, n // 4)
                    low_n = max(1, n // 4)
                    top_group = g_sorted.tail(top_n)
                    low_group = g_sorted.head(low_n)
                    comp_data = []
                    for m in box_cols:
                        comp_data.append({
                            "지표": m,
                            "상위 그룹 평균": round(top_group[m].mean(), 2),
                            "하위 그룹 평균": round(low_group[m].mean(), 2),
                            "차이 (상위-하위)": round(top_group[m].mean() - low_group[m].mean(), 2),
                        })
                    df_comp = pd.DataFrame(comp_data)
                    st.dataframe(df_comp, use_container_width=True, hide_index=True)
                    st.caption(f"상위 {top_n}개체(생중량 최대) vs 하위 {low_n}개체(생중량 최소) 비교")
                else:
                    st.info("📎 개체 수가 부족해 상위·하위 그룹 비교를 할 수 없습니다.")

            # 생육 결과 CSV 다운로드
            if not growth_df.empty:
                csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 생육 데이터 CSV 다운로드 (utf-8-sig)",
                    data=csv_bytes,
                    file_name="송천고_생육결과_분석용.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
'''

with open('/home/user/outputs/main.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("✅ main.py 저장 완료: /home/user/outputs/main.py")
print(f"   길이: {len(code)}자")
print(f"   라인 수: {code.count(chr(10))+1}줄")
