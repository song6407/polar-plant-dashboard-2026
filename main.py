import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="극지식물 생육 분석", layout="wide")
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

@st.cache_data
def read_csv_cached(file):
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

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
                .str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce"
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
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df

def school_name(filename):
    if filename is None:
        return "우리 학교"
    base = os.path.basename(filename)
    name = base.split("_")[0].strip()
    return name if name else "우리 학교"

st.sidebar.header("📁 CSV 업로드")
env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv")

env_raw = read_csv_cached(env_file)
growth_raw = read_csv_cached(growth_file)
env_df = std_cols(env_raw) if not env_raw.empty else pd.DataFrame()
growth_df = std_cols(growth_raw) if not growth_raw.empty else pd.DataFrame()
school = school_name(env_file.name if env_file else None)
st.sidebar.success(f"📍 학교명: {school}")

st.sidebar.subheader("📏 생육 측정 항목 선택")
growth_cols_all = [
    "잎 수", "지상부 길이", "지하부 길이",
    "지상부 질량", "지하부 질량", "생중량",
]
available_cols = [c for c in growth_cols_all if c in growth_df.columns]
selected_cols = st.sidebar.multiselect(
    "비교할 항목 (개체번호 제외)",
    available_cols,
    default=available_cols,
)
st.sidebar.caption("💡 생중량이 없는 학교는 지상부 질량 + 지하부 질량으로 자동 계산됩니다.")

def num_col(df, col):
    if col not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[col], errors="coerce")
    return s.dropna()

def daily_mean(df, col, time_col="time"):
    if col not in df.columns or time_col not in df.columns:
        return pd.DataFrame()
    tmp = df[[time_col, col]].dropna()
    if tmp.empty:
        return pd.DataFrame()
    tmp[time_col] = pd.to_datetime(tmp[time_col])
    tmp["date"] = tmp[time_col].dt.date
    agg = tmp.groupby("date")[col].mean().reset_index()
    agg.columns = ["date", col]
    return agg

tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

with tab1:
    try:
        st.header("🔬 연구 배경 및 목적")
        st.markdown(
            f"**{school}**에서 재배 중인 극지식물의 **개체별 생육 차이**를 분석합니다. "
            "같은 환경에서 자라도 개체마다 잎 수, 줄기 길이, 생체량이 다를 수 있어요. "
            "이 차이를 수치(평균·표준편차·상관계수)로 확인하면 어떤 개체가 잘 자랐는지, "
            "무엇이 생육에 영향을 주는지 알 수 있습니다."
        )
        st.markdown("📌 **통계 용어 참고**")
        st.markdown(
            "- **평균**: 여러 값을 더해 개수로 나눈 대표값\n"
            "- **표준편차**: 값들이 평균에서 얼마나 흩어져 있는지 나타내는 수치 (클수록 들쭉날쭉)\n"
            "- **CV(변동계수)**: 표준편차를 평균으로 나눈 값 (%로 표시) — 평균이 다를 때도 비교 가능\n"
            "- **상관계수(r)**: 두 변수가 함께 움직이는 정도를 -1~1로 나타낸 값 (1에 가까울수록 함께 증가)"
        )
        st.header("📋 데이터 요약")
        if not growth_df.empty:
            n = len(growth_df)
            date_min = ""
            date_max = ""
            if "time" in env_df.columns and not env_df["time"].isna().all():
                date_min = env_df["time"].min().strftime("%Y-%m-%d")
                date_max = env_df["time"].max().strftime("%Y-%m-%d")
            items = []
            if "잎 수" in growth_df.columns:
                items.append("잎 수")
            if "지상부 길이" in growth_df.columns:
                items.append("지상부 길이")
            if "지하부 길이" in growth_df.columns:
                items.append("지하부 길이")
            if "지상부 질량" in growth_df.columns:
                items.append("지상부 질량")
            if "지하부 질량" in growth_df.columns:
                items.append("지하부 질량")
            if "생중량" in growth_df.columns:
                items.append("생중량")
            rows = []
            rows.append(["총 개체 수", f"{n}개체", ""])
            rows.append(["측정 기간", date_min, date_max])
            rows.append(["측정 항목", ", ".join(items) if items else "없음", ""])
            st.table(pd.DataFrame(rows, columns=["항목", "값", "비고"]))
        else:
            st.info("📌 생육 결과 CSV를 업로드하면 데이터 요약이 표시됩니다.")
        st.header("📊 주요 지표")
        col1, col2, col3, col4 = st.columns(4)
        if not growth_df.empty and "생중량" in growth_df.columns:
            gw = num_col(growth_df, "생중량")
            if len(gw) >= 2:
                mean_gw = gw.mean()
                col4.metric("평균 생중량", f"{mean_gw:.2f} g")
        if "temperature" in env_df.columns:
            t = num_col(env_df, "temperature")
            if len(t) >= 1:
                col2.metric("평균 온도", f"{t.mean():.1f} °C")
        if "humidity" in env_df.columns:
            h = num_col(env_df, "humidity")
            if len(h) >= 1:
                col3.metric("평균 습도", f"{h.mean():.1f} %")
        if not growth_df.empty:
            col1.metric("총 개체 수", f"{len(growth_df)} 개체")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab2:
    try:
        if env_df.empty:
            st.info("📌 환경 데이터 CSV를 업로드하면 환경 분석 결과가 표시됩니다.")
        else:
            st.header("🌡️ 환경 요약")
            env_summary_cols = [
                c for c in ["temperature", "humidity", "ph", "ec"]
                if c in env_df.columns
            ]
            if env_summary_cols:
                summary_rows = []
                for col in env_summary_cols:
                    s = num_col(env_df, col)
                    if len(s) >= 1:
                        summary_rows.append([
                            col, f"{s.mean():.2f}", f"{s.min():.2f}", f"{s.max():.2f}"
                        ])
                if summary_rows:
                    st.table(pd.DataFrame(
                        summary_rows,
                        columns=["항목", "평균", "최소", "최대"]
                    ))
                else:
                    st.info("📌 환경 수치가 충분하지 않아 요약표를 만들 수 없습니다.")
            else:
                st.info("📌 온도·습도·pH·EC 열이 모두 없어 환경 요약을 표시할 수 없습니다.")
            st.header("📈 날짜별 환경 변화 (일별 평균)")
            env_ts_cols = [
                c for c in ["temperature", "humidity", "ph", "ec"]
                if c in env_df.columns
            ]
            if env_ts_cols:
                n_ts = len(env_ts_cols)
                ts_fig = make_subplots(
                    rows=n_ts, cols=1,
                    subplot_titles=[
                        "온도 변화" if c == "temperature" else
                        "습도 변화" if c == "humidity" else
                        "pH 변화" if c == "ph" else
                        "EC 변화"
                        for c in env_ts_cols
                    ],
                    vertical_spacing=0.12,
                )
                for i, col in enumerate(env_ts_cols):
                    dm = daily_mean(env_df, col)
                    if dm.empty:
                        continue
                    s = num_col(dm, col)
                    if len(s) < 2:
                        continue
                    day_labels = [str(d) for d in dm["date"]]
                    ts_fig.add_trace(
                        go.Scatter(
                            x=day_labels, y=s.values,
                            mode="lines",
                            name=col,
                            line=dict(width=2),
                        ),
                        row=i + 1, col=1,
                    )
                if ts_fig.data:
                    ts_fig.update_layout(
                        height=280 * n_ts + 40,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        showlegend=False,
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                    for i in range(n_ts):
                        ts_fig.update_yaxes(title_text="", row=i + 1, col=1)
                    st.plotly_chart(ts_fig, width="stretch")
                else:
                    st.info("📌 일별 평균을 계산할 데이터가 충분하지 않습니다.")
            else:
                st.info("📌 온도·습도·pH·EC 열이 없어 시계열 그래프를 표시할 수 없습니다.")
            with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(env_df, width="stretch")
                csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school}_환경데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab3:
    try:
        if growth_df.empty:
            st.info("📌 생육 결과 CSV를 업로드하면 생육 분석 결과가 표시됩니다.")
        else:
            st.header("🥇 핵심 결과")
            kcard1, kcard2, kcard3 = st.columns(3)
            if "생중량" in growth_df.columns:
                gw = num_col(growth_df, "생중량")
                if len(gw) >= 2:
                    mean_gw = gw.mean()
                    max_idx = gw.idxmax()
                    min_idx = gw.idxmin()
                    kcard1.metric(
                        "평균 생중량",
                        f"{mean_gw:.2f} g",
                        delta=f"CV {gw.std()/gw.mean()*100:.1f}%" if gw.mean() > 0 else None,
                    )
                    kcard2.metric(
                        "최대 생중량 개체",
                        f"{growth_df.loc[max_idx, '개체번호']}",
                        delta=f"{gw.max():.2f} g",
                    )
                    kcard3.metric(
                        "최소 생중량 개체",
                        f"{growth_df.loc[min_idx, '개체번호']}",
                        delta=f"{gw.min():.2f} g",
                    )
            st.header("📊 개체별 생육 비교")
            if selected_cols:
                bar_cols = [c for c in selected_cols if c in growth_df.columns]
                if bar_cols:
                    def make_bar(col, title):
                        s = num_col(growth_df, col)
                        if len(s) < 2:
                            return None
                        idx_sorted = s.sort_values(ascending=False).index
                        top5 = idx_sorted[:5]
                        bot5 = idx_sorted[-5:]
                        colors = []
                        for idx in s.index:
                            if idx in top5:
                                colors.append("#2ecc71")
                            elif idx in bot5:
                                colors.append("#e74c3c")
                            else:
                                colors.append("#95a5a6")
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=growth_df.loc[s.index, "개체번호"].values,
                            y=s.values,
                            marker_color=colors,
                            marker_line_width=0,
                        ))
                        fig.update_layout(
                            title=title,
                            xaxis_title="개체번호",
                            yaxis_title=col,
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            margin=dict(l=40, r=20, t=40, b=40),
                            height=380,
                        )
                        return fig
                    bc1, bc2 = st.columns(2 if len(bar_cols) <= 2 else 1)
                    for i, col in enumerate(bar_cols):
                        fig = make_bar(col, f"개체별 {col}")
                        if fig is not None:
                            if len(bar_cols) <= 2:
                                bc = bc1 if i == 0 else bc2
                                with bc:
                                    st.plotly_chart(fig, width="stretch")
                            else:
                                st.plotly_chart(fig, width="stretch")
                else:
                    st.info("📌 선택한 측정 항목이 데이터에 없습니다.")
            else:
                st.info("📌 비교할 측정 항목을 사이드바에서 선택하세요.")
            if "생중량" in growth_df.columns:
                gw = num_col(growth_df, "생중량")
                if len(gw) >= 4:
                    st.header("📦 생중량 분포")
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=gw.values,
                        name="생중량",
                        fillcolor="#3498db",
                        line_color="#2980b9",
                        boxpoints="outliers",
                        marker=dict(color="#2980b9", size=6),
                        jitter=0.3,
                    ))
                    fig_box.update_layout(
                        title="개체별 생중량 분포 (박스플롯)",
                        yaxis_title="생중량 (g)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        margin=dict(l=40, r=20, t=40, b=40),
                        height=380,
                    )
                    st.plotly_chart(fig_box, width="stretch")
                    st.header("🏆 생중량 상위 25% vs 하위 25% 비교")
                    n = len(gw)
                    q1_idx = int(n * 0.25)
                    q3_idx = int(n * 0.75)
                    gw_sorted = gw.sort_values()
                    top25 = gw_sorted.iloc[q3_idx:]
                    bot25 = gw_sorted.iloc[:q1_idx]
                    comp1, comp2, comp3 = st.columns(3)
                    comp1.metric("상위 25% 평균", f"{top25.mean():.2f} g", f"{len(top25)}개체")
                    comp2.metric("하위 25% 평균", f"{bot25.mean():.2f} g", f"{len(bot25)}개체")
                    ratio = top25.mean() / bot25.mean() if bot25.mean() > 0 else 0
                    comp3.metric("배율 (상위/하위)", f"{ratio:.2f}배")
                    st.caption("💡 상위 25%는 생중량이 가장 큰 개체들, 하위 25%는 가장 작은 개체들입니다.")
                else:
                    st.info("📌 생중량 데이터가 충분하지 않아 분포 그래프를 표시할 수 없습니다.")
            st.header("🔗 상관관계 분석")
            corr_cols = [c for c in ["잎 수", "지상부 길이"] if c in growth_df.columns]
            if "생중량" in growth_df.columns and corr_cols:
                n_corr = len(corr_cols)
                corr_fig = make_subplots(
                    rows=n_corr, cols=1,
                    subplot_titles=[f"{x} vs 생중량" for x in corr_cols],
                    vertical_spacing=0.12,
                )
                for i, x_col in enumerate(corr_cols):
                    x = num_col(growth_df, x_col)
                    y = num_col(growth_df, "생중량")
                    df_xy = pd.DataFrame({"x": x.values, "y": y.values}).dropna()
                    if len(df_xy) < 3:
                        continue
                    xx = df_xy["x"].values
                    yy = df_xy["y"].values
                    r, p = stats.pearsonr(xx, yy)
                    slope, intercept = np.polyfit(xx, yy, 1)
                    line_y = slope * xx + intercept
                    corr_fig.add_trace(
                        go.Scatter(
                            x=xx, y=yy,
                            mode="markers",
                            marker=dict(size=9, color="#3498db", line=dict(width=1, color="white")),
                            name=f"{x_col} vs 생중량",
                            text=growth_df.loc[x.index, "개체번호"].values[:len(xx)],
                        ),
                        row=i + 1, col=1,
                    )
                    corr_fig.add_trace(
                        go.Scatter(
                            x=xx, y=line_y,
                            mode="lines",
                            line=dict(color="#e74c3c", width=2, dash="dash"),
                            name=f"추세선 (r={r:.3f})",
                        ),
                        row=i + 1, col=1,
                    )
                if corr_fig.data:
                    corr_fig.update_layout(
                        height=340 * n_corr + 40,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        showlegend=False,
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                    for i in range(n_corr):
                        corr_fig.update_xaxes(title_text=corr_cols[i], row=i + 1, col=1)
                        corr_fig.update_yaxes(title_text="생중량 (g)", row=i + 1, col=1)
                    st.plotly_chart(corr_fig, width="stretch")
                    st.caption(
                        "📌 **상관계수(r)**가 1에 가까우면 두 값이 함께 증가하고, "
                        "-1에 가까우면 한쪽이 증가할 때 다른 쪽이 감소합니다. "
                        "0에 가까우면 뚜렷한 관계가 없습니다."
                    )
                else:
                    st.info("📌 상관계수를 계산할 데이터가 충분하지 않습니다.")
            else:
                st.info("📌 잎 수 또는 지상부 길이, 생중량 열이 없어 상관관계 분석을 표시할 수 없습니다.")
            with st.expander("📄 생육 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(growth_df, width="stretch")
                csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school}_생육결과데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
