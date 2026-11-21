
import io
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── CSS: 한글 폰트 (Streamlit 한글 깨짐 방지) ─────────────────────
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

# ── CSV 읽기 (utf-8-sig → cp949 2단계) ────────────────────────────
def read_csv(file):
    """업로드된 CSV를 utf-8-sig로 읽고 실패 시 cp949로 재시도."""
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ── 열 이름 표준화 (학교마다 다른 열 이름 → KeyError 방지) ────────
def std_cols(df):
    """학교마다 다른 열 이름을 표준 이름으로 통일."""
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
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(
                parts[1].astype(int), unit="h"
            )
        df["time"] = parsed
    return df

# ── 학교명 추출 (파일명 앞부분, 하드코딩 금지) ─────────────────────
def extract_school(fname):
    """파일명에서 학교명 추출(예: 인천송천고_환경데이터.csv→인천송천고)."""
    base = os.path.splitext(str(fname))[0]
    parts = base.split("_")
    return parts[0] if parts else base

# ── Pearson 상관계수 helper (결측치 제거, 2개 미만이면 None) ────
def safe_corr(x, y):
    """두 series의 Pearson r과 p-value. 2개 미만이면 None."""
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 2:
        return None, None
    r, p = stats.pearsonr(valid.iloc[:, 0].values, valid.iloc[:, 1].values)
    return round(r, 3), round(p, 4)

# ── np.polyfit 추세선 (px.scatter trendline 대신) ──────────────────
def line_trend(fig, x, y, color="grey", dash="dash", name="추세선", row=1, col=1):
    """np.polyfit으로 기울기를 구해 go.Scatter 추세선 추가."""
    clean = pd.concat([x, y], axis=1).dropna()
    if len(clean) < 2:
        return
    xs, ys = clean.iloc[:, 0].values, clean.iloc[:, 1].values
    slope, intercept = np.polyfit(xs, ys, 1)
    fig.add_trace(
        go.Scatter(
            x=np.array([xs.min(), xs.max()]),
            y=slope * np.array([xs.min(), xs.max()]) + intercept,
            mode="lines",
            line=dict(color=color, dash=dash),
            name=name,
        ),
        row=row,
        col=col,
    )

# ── 숫자 지표의 단위 반환 helper ──────────────────────────────────
def unit_of(col):
    """지표 이름에 따라 단위 문자열 반환."""
    if col == "잎 수":
        return "장"
    if "길이" in col:
        return "mm"
    if "질량" in col or col == "생중량":
        return "g"
    return ""

# ═══════════════════════════════════════════════════════════════════
#  탭 1: 실험 목적 & 주요 결과
# ═══════════════════════════════════════════════════════════════════
def metric_card(st_obj, label, value, info_msg=None):
    """st.metric + 선택적 st.info 한 세트."""
    st_obj.metric(label, value)
    if info_msg:
        st_obj.info(info_msg)

def render_info_tab(env_df, growth_df):
    st.title("🔬 실험 목적과 주요 결과")
    school_e = extract_school(st.session_state.get("env_fname", "환경 데이터"))
    school_g = extract_school(st.session_state.get("growth_fname", "생육 데이터"))

    st.subheader("🎯 실험 목적")
    st.markdown(
        """
이 대시보드는 **극지식물의 지하부 생장이 지상부 생장에 어떤 영향을 미치는지** 알아보기 위해 만들었어요.
잎 수·지상부 길이 중 생중량과 더 밀접한 지표를 찾고, 무거운 개체와 가벼운 개체의 생육 특징을 비교합니다.
"""
    )

    st.subheader("📊 핵심 숫자")
    kc1, kc2 = st.columns(2)

    with kc1:
        st.caption(f"환경 데이터: {school_e}")
        if env_df.empty:
            st.info("📁 환경 CSV를 업로드하면 핵심 숫자가 표시됩니다.")
        else:
            timec = "time" if "time" in env_df.columns else None
            for key, col, unit, label in [
                ("temperature", "temperature", "℃", "온도"),
                ("humidity", "humidity", "%", "습도"),
                ("ph", "ph", "", "pH"),
                ("ec", "ec", " mS/cm", "EC"),
            ]:
                if col not in env_df.columns:
                    st.info(f"{label} 열이 없어 평균을 표시할 수 없어요.")
                    continue
                s = env_df[col].dropna()
                if not s.empty:
                    if key in ("temperature", "humidity") and timec:
                        t = env_df[[timec, col]].dropna()
                        if not t.empty:
                            st.metric(f"평균 {label}", f"{t[col].mean():.1f}{unit}")
                            if key == "temperature":
                                st.metric(f"최고 {label}", f"{t[col].max():.1f}{unit}")
                                st.metric(f"최저 {label}", f"{t[col].min():.1f}{unit}")
                            continue
                    st.metric(f"평균 {label}", f"{s.mean():.2f}{unit}")
                else:
                    st.info(f"{label} 데이터가 충분하지 않아요.")

    with kc2:
        st.caption(f"생육 데이터: {school_g}")
        if growth_df.empty:
            st.info("📁 생육 결과 CSV를 업로드하면 핵심 숫자가 표시됩니다.")
        else:
            ids = [str(x) for x in growth_df["개체번호"].tolist()]
            st.metric("개체 수", len(ids))
            st.caption(", ".join(ids[:5]) + (" ... " if len(ids) > 5 else ""))
            for col in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]:
                if col not in growth_df.columns:
                    st.info(f"{col} 열이 없어 평균을 표시할 수 없어요.")
                    continue
                s = growth_df[col].dropna()
                if not s.empty:
                    st.metric(f"{col} 평균", f"{s.mean():.2f} {unit_of(col)}")
                else:
                    st.info(f"{col} 데이터가 충분하지 않아요.")

    st.subheader("💡 전하고 싶은 내용")
    st.markdown(
        """
**지하부 생장이 지상부 생장에 영향을 미친다.**

극지식물은 짧은 생장 기간 동안 뿌리를 먼저 발달시켜 수분과 양분을 확보한 뒤,
그 에너지를 지상부(줄기·잎) 생장으로 전환하는 전략을 취해요.
이 대시보드에서 지하부 길이-지상부 길이 간 상관관계, 무거운/가벼운 개체 차이, 생중량과 가장 밀접한 지표를 확인할 수 있어요.
"""
    )

# ═══════════════════════════════════════════════════════════════════
#  탭 2: 날짜별 환경 변화
# ═══════════════════════════════════════════════════════════════════
def render_env_tab(env_df):
    school = extract_school(st.session_state.get("env_fname", "환경 데이터"))
    st.title("📅 날짜별 환경 변화")
    st.caption(f"데이터 출처: {school}")

    cols_avail = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]; colors_map = {"temperature": "#e74c3c", "humidity": "#3498db", "ph": "#2ecc71", "ec": "#f39c12"}

    if not cols_avail:
        st.info("📈 꺾은선 그래프로 표시할 환경 지표(온도·습도·pH·EC)가 없습니다.")
    else:
        try:
            with st.spinner("환경 그래프 그리는 중..."):
                fig = make_subplots(rows=len(cols_avail), cols=1, subplot_titles=cols_avail, shared_xaxes=True)
                for i, col in enumerate(cols_avail, 1):
                    s = env_df[["time", col]].dropna()
                    if s.empty or len(s) < 2:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=s["time"], y=s[col], mode="lines", name=col,
                            line=dict(color=colors_map.get(col, "#333")),
                            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}<extra></extra>",
                        ),
                        row=i, col=1,
                    )
                fig.update_layout(
                    height=200 * len(cols_avail),
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    showlegend=False, hovermode="x unified",
                )
                st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    st.subheader("📋 환경 데이터 원본")
    disp = env_df.copy()
    if "time" in disp.columns:
        disp["time"] = disp["time"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(disp, width="stretch", use_container_width=False)
    st.download_button(
        "환경 데이터 CSV 다운로드", disp.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{school}_환경데이터.csv", mime="text/csv",
    )

# ═══════════════════════════════════════════════════════════════════
#  탭 3: 개체별 생육 분석
# ═══════════════════════════════════════════════════════════════════
def render_growth_tab(growth_df):
    school = extract_school(st.session_state.get("growth_fname", "생육 데이터"))
    st.title("🌱 개체별 생육 분석")
    st.caption(f"데이터 출처: {school}")

    num_cols = [
        c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"]
        if c in growth_df.columns
    ]
    if not num_cols:
        st.info("📊 그래프로 표시할 생육 지표(잎 수, 길이, 질량)가 없습니다.")
        return

    id_col = "개체번호"; top_n = st.sidebar.slider("상·하위 개체 수", 3, 15, 5, step=1, key="topn")

    # ── 개체별 막대 그래프 ─────────────────────────────────────────
    try:
        with st.spinner("생육 막대 그래프 그리는 중..."):
            fig_bar = make_subplots(
                rows=len(num_cols), cols=1, subplot_titles=num_cols,
                shared_xaxes=False, vertical_spacing=0.08,
            )
            pd_sorted = growth_df[[id_col] + num_cols].copy()
            for ridx, col in enumerate(num_cols, 1):
                s = pd_sorted.dropna(subset=[col]).sort_values(col, ascending=False)
                top_ids, bot_ids = s[id_col].head(top_n).tolist(), s[id_col].tail(top_n).tolist()
                bar_colors = ["#e74c3c" if idx in top_ids else ("#3498db" if idx in bot_ids else "#95a5a6") for idx in s[id_col]]
                fig_bar.add_trace(go.Bar(x=s[id_col], y=s[col], marker=dict(color=bar_colors[: len(s)]), name=col, hovertemplate="%{x}<br>%{y:.2f}<extra></extra>"), row=ridx, col=1)
                fig_bar.update_yaxes(title_text=f"{col} ({unit_of(col)})", row=ridx, col=1)
            fig_bar.update_layout(
                height=200 * len(num_cols) + 50,
                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                showlegend=False, hovermode="x unified",
            )
            st.plotly_chart(fig_bar, width="stretch")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    # ── 지하부 vs 지상부 산점도 ─────────────────────────────────────
    try:
        if "지하부 길이" in growth_df.columns and "지상부 길이" in growth_df.columns:
            s_all = growth_df[["지하부 길이", "지상부 길이", "개체번호"]].dropna()
            if len(s_all) >= 2:
                with st.spinner("산점도 그리는 중..."):
                    fig_s = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=["지하부 vs 지상부 길이 (전체)", "지하부 vs 지상부 길이 (생중량 상위/하위)"],
                    )
                    # 왼쪽: 전체
                    fig_s.add_trace(go.Scatter(x=s_all["지하부 길이"], y=s_all["지상부 길이"], mode="markers", marker=dict(size=7, color="#2ecc71", opacity=0.7), hovertemplate="지상부 %{y:.1f}mm<br>지하부 %{x:.1f}mm<extra></extra>"), row=1, col=1)
                    line_trend(fig_s, s_all["지하부 길이"], s_all["지상부 길이"])
                    r_all, _ = safe_corr(s_all["지하부 길이"], s_all["지상부 길이"])
                    fig_s.update_xaxes(title_text="지하부 길이 (mm)", row=1, col=1)
                    fig_s.update_yaxes(title_text="지상부 길이 (mm)", row=1, col=1)
                    if r_all is not None:
                        fig_s.add_annotation(
                            text=f"전체 r = {r_all}", xref="paper", yref="paper",
                            x=0.95, y=0.05, showarrow=False, font=dict(size=13, color="#333"),
                            row=1, col=1,
                        )
                    # 오른쪽: 무거운/가벼운 그룹
                    if "생중량" in growth_df.columns:
                        med = growth_df["생중량"].median()
                        heavy_ids = growth_df[growth_df["생중량"] >= med]["개체번호"].tolist()
                        light_ids = growth_df[growth_df["생중량"] < med]["개체번호"].tolist()
                        hc = growth_df[growth_df["개체번호"].isin(heavy_ids)][["지하부 길이", "지상부 길이"]].dropna(); lc = growth_df[growth_df["개체번호"].isin(light_ids)][["지하부 길이", "지상부 길이"]].dropna()
                        if not hc.empty:
                            fig_s.add_trace(go.Scatter(x=hc["지하부 길이"], y=hc["지상부 길이"], mode="markers", marker=dict(size=9, color="#e74c3c"), name="생중량 무거운 그룹", hovertemplate="지상부 %{y:.1f}mm<br>지하부 %{x:.1f}mm<extra></extra>"), row=1, col=2)
                        if not lc.empty:
                            fig_s.add_trace(go.Scatter(x=lc["지하부 길이"], y=lc["지상부 길이"], mode="markers", marker=dict(size=9, color="#3498db"), name="생중량 가벼운 그룹", hovertemplate="지상부 %{y:.1f}mm<br>지하부 %{x:.1f}mm<extra></extra>"), row=1, col=2)
                        if len(hc) >= 2:
                            line_trend(fig_s, hc["지하부 길이"], hc["지상부 길이"], color="red", name="무거운 그룹 추세선", row=1, col=2)
                        if len(lc) >= 2:
                            line_trend(fig_s, lc["지하부 길이"], lc["지상부 길이"], color="blue", name="가벼운 그룹 추세선", row=1, col=2)
                        r_h, _ = (safe_corr(hc["지하부 길이"], hc["지상부 길이"]) if not hc.empty else (None, None))
                        r_l, _ = (safe_corr(lc["지하부 길이"], lc["지상부 길이"]) if not lc.empty else (None, None))
                        if r_h is not None:
                            fig_s.add_annotation(text=f"무거운 그룹 r = {r_h}", xref="paper", yref="paper",
                                x=0.95, y=0.05, showarrow=False, font=dict(size=11, color="#e74c3c"), row=1, col=2)
                        if r_l is not None:
                            fig_s.add_annotation(text=f"가벼운 그룹 r = {r_l}", xref="paper", yref="paper",
                                x=0.95, y=0.15, showarrow=False, font=dict(size=11, color="#3498db"), row=1, col=2)
                    fig_s.update_xaxes(title_text="지하부 길이 (mm)", row=1, col=2)
                    fig_s.update_yaxes(title_text="지상부 길이 (mm)", row=1, col=2)
                    fig_s.update_layout(
                        height=480,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        showlegend=True, hovermode="closest",
                    )
                    st.plotly_chart(fig_s, width="stretch")
            else:
                st.info("산점도를 그리기에 데이터가 부족합니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    # ── 무거운 개체 vs 가벼운 개체 비교 표 ────────────────────────
    try:
        if "생중량" in growth_df.columns:
            med = growth_df["생중량"].median()
            heavy = growth_df[growth_df["생중량"] >= med]
            light = growth_df[growth_df["생중량"] < med]
            st.subheader("📊 무거운 개체 vs 가벼운 개체 비교")
            c1, c2 = st.columns(2)
            c1.metric("무거운 개체 수 (중앙값 이상)", len(heavy))
            c2.metric("가벼운 개체 수 (중앙값 미만)", len(light))
            rows = []
            for col in num_cols:
                hv, lv = heavy[col].dropna().mean(), light[col].dropna().mean()
                if pd.notna(hv) and pd.notna(lv):
                    rows.append([col, f"{hv:.2f}", f"{lv:.2f}", f"{hv - lv:+.2f}"])
            if rows:
                st.dataframe(
                    pd.DataFrame(rows, columns=["지표", "무거운 개체 평균", "가벼운 개체 평균", "차이(무거움−가벼움)"]),
                    width="stretch", use_container_width=False,
                )
            else:
                st.info("비교할 지표가 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 비교 표 표시 중 오류가 있었어요: {e}")

    # ── 생중량과 가장 밀접한 지표 찾기 ────────────────────────────
    try:
        candidates = [c for c in num_cols if c != "생중량"]
        if candidates:
            target = st.sidebar.selectbox(
                "생중량과 더 밀접한 지표 찾기", candidates,
                index=0 if "잎 수" in candidates else 0, key="corr_target",
            )
            st.subheader("🔍 생중량과 생육 지표의 상관관계")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"#### {target} vs 생중량")
                r_val, p_val = safe_corr(growth_df[target], growth_df["생중량"])
                if r_val is not None:
                    st.metric("Pearson r", r_val)
                    st.metric("p-value", p_val)
                    strength = "강한 상관" if abs(r_val) >= 0.7 else ("중간 상관" if abs(r_val) >= 0.4 else "약한 상관")
                    direction = "양의 상관" if r_val > 0 else "음의 상관"
                    st.info(f"{direction}, {strength} (|r| = {abs(r_val):.3f})")
                else:
                    st.info("상관 계수를 계산하기에 데이터가 부족합니다.")
            with c2:
                st.markdown("#### 생중량과 가장 밀접한 지표")
                best_r, best_name = 0.0, "없음"
                for c in candidates:
                    rv, _ = safe_corr(growth_df[c], growth_df["생중량"])
                    if rv is not None and abs(rv) > abs(best_r):
                        best_r, best_name = rv, c
                if best_name != "없음":
                    st.success(f"가장 높은 상관: **{best_name}** (r = {best_r:.3f})")
                else:
                    st.info("비교할 지표가 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 상관 분석 중 오류가 있었어요: {e}")

    # ── 통계 용어 간단 설명 ───────────────────────────────────────
    try:
        st.subheader("📝 통계 용어 간단 설명")
        st.markdown(
            """
- **평균**: 모든 값을 더해 개수로 나눈 대표값.
- **표준편차**: 값들이 평균에서 얼마나 퍼져 있는지 나타냄. 클수록 개체 간 차이가 큼.
- **Pearson r**: 두 지표의 함께 움직이는 정도 (−1 ~ +1). +1에 가까우면 함께 증가, −1에 가까우면 반대 방향.
- **p-value**: 우연일 가능성. 보통 0.05 미만이면 통계적으로 의미 있음.
- **중앙값**: 크기순 정 가운데 값. 극단적 값의 영향을 덜 받음.
            """
        )
    except Exception:
        pass

# ── 사이드바: CSV 업로드 & 측정 항목 선택 ─────────────────────────
st.sidebar.header("📂 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 CSV", type="csv", key="env_uploader")
growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv", key="growth_uploader")

env_df, growth_df = pd.DataFrame(), pd.DataFrame()
if env_file is not None:
    with st.spinner("환경 데이터 읽는 중..."):
        env_df = std_cols(read_csv(env_file))
        st.session_state["env_fname"] = env_file.name
if growth_file is not None:
    with st.spinner("생육 결과 데이터 읽는 중..."):
        growth_df = std_cols(read_csv(growth_file))
        st.session_state["growth_fname"] = growth_file.name

st.sidebar.header("📐 생육 측정 항목 선택")
growth_cols_all = ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"]
if not growth_df.empty:
    available = [c for c in growth_cols_all if c in growth_df.columns]
    st.sidebar.multiselect("그래프에 표시할 생육 지표", available, default=available, key="growth_select")
else:
    st.sidebar.info("생육 CSV를 먼저 업로드하세요.")

# ── 탭 구성 ────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔬 실험 목적 & 주요 결과", "📅 날짜별 환경 변화", "🌱 개체별 생육 분석"])

with tab1:
    try:
        render_info_tab(env_df, growth_df)
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab2:
    try:
        if env_df.empty:
            st.info("📁 환경 CSV를 업로드하면 날짜별 환경 변화 그래프와 원본 표가 표시됩니다.")
        else:
            render_env_tab(env_df)
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab3:
    try:
        if growth_df.empty:
            st.info("📁 생육 결과 CSV를 업로드하면 개체별 생육 그래프와 분석이 표시됩니다.")
        else:
            render_growth_tab(growth_df)
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
