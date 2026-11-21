import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
font_path = None
for f in fm.findSystemFonts():
    if 'noto' in f.lower() and 'kr' in f.lower():
        font_path = f
        break
if not font_path:
    for f in fm.findSystemFonts():
        if 'nanum' in f.lower():
            font_path = f
            break
if not font_path:
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    try:
        fm.FontProperties(fname=font_path)
    except:
        font_path = None

if font_path:
    fp = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = fp.get_name()
    print(f"사용 폰트: {fp.get_name()} ({font_path})")
else:
    plt.rcParams['font.family'] = 'sans-serif'
    print("폰트 찾을 수 없음 — sans-serif 사용")

plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
env_path = '/home/user/uploads/0____________________________.csv'
growth_path = '/home/user/uploads/1______________.csv'

with open(env_path, 'rb') as f:
    env_df = pd.read_csv(f, encoding='utf-8-sig')
with open(growth_path, 'rb') as f:
    growth_df = pd.read_csv(f, encoding='utf-8-sig')

# 열 이름 표준화
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time","categories","시간","날짜","datetime","date"):
            rename[c] = "time"
        elif k in ("temperature","온도"):
            rename[c] = "temperature"
        elif k in ("humidity","습도"):
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
    if "생중량" not in df.columns and {"지상부 질량","지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df

env_df = std_cols(env_df)
growth_df = std_cols(growth_df)

print(f"\n환경 데이터: {env_df.shape}, 열: {list(env_df.columns)}")
print(f"생육 데이터: {growth_df.shape}, 열: {list(growth_df.columns)}")

# ─── Figure 1: 날짜별 온도·습도 변화 ───
fig1, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig1.patch.set_facecolor('#f8fafc')

# 온도
t = env_df['time'].dropna()
temp = env_df['temperature'].dropna()
mask_t = temp.index.intersection(t.index)
ax1 = axes[0]
ax1.plot(t.loc[mask_t], temp.loc[mask_t], color='#f97316', linewidth=1.2, label='온도')
ax1.fill_between(t.loc[mask_t], temp.loc[mask_t], alpha=0.15, color='#f97316')
ax1.axhline(temp.mean(), color='#dc2626', linestyle='--', linewidth=1, label=f'평균 {temp.mean():.1f}°C')
ax1.set_ylabel('온도 (°C)', fontsize=11, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.set_title('🌡️ 날짜별 온도 변화 (2026-07-15 ~ 2026-10-15)', fontsize=13, fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 35)

# 습도
hum = env_df['humidity'].dropna()
mask_h = hum.index.intersection(t.index)
ax2 = axes[1]
ax2.plot(t.loc[mask_h], hum.loc[mask_h], color='#3b82f6', linewidth=1.2, label='습도')
ax2.fill_between(t.loc[mask_h], hum.loc[mask_h], alpha=0.15, color='#3b82f6')
ax2.axhline(hum.mean(), color='#1d4ed8', linestyle='--', linewidth=1, label=f'평균 {hum.mean():.1f}%')
ax2.set_ylabel('습도 (%)', fontsize=11, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.set_title('💧 날짜별 습도 변화', fontsize=13, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 100)

fig1.tight_layout()
fig1.savefig('/home/user/outputs/fig1_env_timeseries.png', dpi=120, bbox_inches='tight', facecolor='#f8fafc')
print("\n✅ Figure 1 저장 완료: 환경 시계열")
plt.close(fig1)

# ─── Figure 2: 개체별 생중량 막대 그래프 ───
fig2, ax = plt.subplots(figsize=(14, 6))
fig2.patch.set_facecolor('#f8fafc')
ax.set_facecolor('#ffffff')

g = growth_df[['개체번호','생중량']].dropna().sort_values('생중량', ascending=True).reset_index(drop=True)
n = len(g)
colors = []
for i in range(n):
    if i < max(1, n//4):
        colors.append('#10b981')  # 상위 25% - 초록
    elif i >= n - max(1, n//4):
        colors.append('#f59e0b')  # 하위 25% - 주황
    else:
        colors.append('#94a3b8')  # 중간 - 회색

bars = ax.bar(range(n), g['생중량'], color=colors, edgecolor='white', linewidth=0.8)
ax.set_xticks(range(n))
ax.set_xticklabels(g['개체번호'], rotation=90, fontsize=7)
ax.set_ylabel('생중량 (g)', fontsize=12, fontweight='bold')
ax.set_xlabel('개체번호', fontsize=12, fontweight='bold')
ax.set_title(f'🌿 개체별 생중량 (총 {n}개체)  |  초록=상위 25%  주황=하위 25%', fontsize=13, fontweight='bold', pad=12)
ax.grid(axis='y', alpha=0.3)
ax.axhline(g['생중량'].mean(), color='#6366f1', linestyle='--', linewidth=1.5, label=f'평균 {g["생중량"].mean():.2f}g')
ax.legend(fontsize=9)

# 값 레이블 (상위·하위만)
for i, (idx, row) in enumerate(g.iterrows()):
    if colors[i] != '#94a3b8':
        ax.text(i, row['생중량'] + 0.02, f"{row['생중량']:.2f}", ha='center', va='bottom', fontsize=7, fontweight='bold')

fig2.tight_layout()
fig2.savefig('/home/user/outputs/fig2_growth_bar.png', dpi=120, bbox_inches='tight', facecolor='#f8fafc')
print("✅ Figure 2 저장 완료: 개체별 생중량 막대")
plt.close(fig2)

# ─── Figure 3: 지하부 길이 vs 지상부 길이 산점도 + 추세선 ───
fig3, ax = plt.subplots(figsize=(9, 7))
fig3.patch.set_facecolor('#f8fafc')
ax.set_facecolor('#ffffff')

xcol, ycol = '지하부 길이', '지상부 길이'
gx = growth_df[xcol].dropna()
gy = growth_df[ycol].dropna()
mask = gx.index.intersection(gy.index)
x = gx.loc[mask].values
y = gy.loc[mask].values

r = np.corrcoef(x, y)[0, 1]
slope, intercept = np.polyfit(x, y, 1)

# 배경 포인트
ax.scatter(x, y, s=60, color='#2d6a9f', alpha=0.6, edgecolors='white', linewidth=0.8, zorder=3, label=f'개체 (n={len(x)})')

# 추세선
x_line = np.linspace(x.min(), x.max(), 100)
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, color='#ef4444', linewidth=2.5, linestyle='--', label=f'추세선 (r={r:.3f}, 기울기={slope:.2f})', zorder=2)

# 강조: 상위 3개체
top3_idx = g['생중량'].nlargest(3).index
for i in top3_idx:
    if i in mask:
        ax.scatter(x[np.where(mask == i)[0][0]], y[np.where(mask == i)[0][0]], s=180, color='#10b981', edgecolors='#047857', linewidth=2, zorder=4, label='생중량 상위 3')

# 강조: 하위 3개체
bot3_idx = g['생중량'].nsmallest(3).index
for i in bot3_idx:
    if i in mask:
        ax.scatter(x[np.where(mask == i)[0][0]], y[np.where(mask == i)[0][0]], s=180, color='#f97316', edgecolors='#c2410c', linewidth=2, zorder=4, label='생중량 하위 3')

ax.set_xlabel('지하부 길이 (mm)', fontsize=12, fontweight='bold')
ax.set_ylabel('지상부 길이 (mm)', fontsize=12, fontweight='bold')
ax.set_title(f'🔬 지하부 길이 vs 지상부 길이  (상관계수 r = {r:.3f})', fontsize=14, fontweight='bold', pad=14)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

# 통계 박스
stats_text = (
    f'n = {len(x)}개체\n'
    f'r = {r:.3f}\n'
    f'기울기 = {slope:.3f}\n'
    f'절편 = {intercept:.2f}\n'
    f'해석: 지하부가 1mm 증가할 때마다\n'
    f'     지상부는 약 {slope:.2f}mm 증가'
)
ax.text(0.97, 0.03, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#eef2ff', edgecolor='#6366f1', alpha=0.9),
        family='monospace')

fig3.tight_layout()
fig3.savefig('/home/user/outputs/fig3_scatter_trend.png', dpi=120, bbox_inches='tight', facecolor='#f8fafc')
print("✅ Figure 3 저장 완료: 산점도 + 추세선")
plt.close(fig3)

# ─── Figure 4: 생육 지표 통계 요약 카드-style 막대 ───
fig4, axes = plt.subplots(1, 3, figsize=(15, 5))
fig4.patch.set_facecolor('#f8fafc')
metrics = [
    ('잎 수', '잎 수', '#8b5cf6'),
    ('지상부 길이', '지상부 길이', '#3b82f6'),
    ('지하부 길이', '지하부 길이', '#10b981'),
]
for ax, (title, col, color) in zip(axes, metrics):
    if col in growth_df.columns:
        s = growth_df[col].dropna()
        ax.set_facecolor('#ffffff')
        bp = ax.boxplot(s, vert=True, patch_artist=True, widths=0.5,
                        boxprops=dict(facecolor=color, alpha=0.4, edgecolor=color),
                        medianprops=dict(color='#1e293b', linewidth=2),
                        whiskerprops=dict(color='#94a3b8'),
                        capprops=dict(color='#94a3b8'))
        ax.scatter(np.ones(len(s)) * 1.1, s, color=color, alpha=0.5, s=20, zorder=2)
        ax.set_title(f'{title}\n(평균 {s.mean():.1f}, 범위 {s.min():.0f}~{s.max():.0f})', fontsize=11, fontweight='bold')
        ax.set_xticklabels([title])
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center', fontsize=12, color='#94a3b8', transform=ax.transAxes)

fig4.suptitle(f'📊 {growth_df.shape[0]}개체 생육 지표 분포', fontsize=14, fontweight='bold', y=1.02)
fig4.tight_layout()
fig4.savefig('/home/user/outputs/fig4_metric_boxplot.png', dpi=120, bbox_inches='tight', facecolor='#f8fafc')
print("✅ Figure 4 저장 완료: 생육 지표 분포")
plt.close(fig4)

print("\n" + "="*50)
print("모든 시각물 저장 완료!")
print("저장 위치: /home/user/outputs/")
import os
for f in sorted(os.listdir('/home/user/outputs')):
    if f.endswith('.png'):
        sz = os.path.getsize(f'/home/user/outputs/{f}')
        print(f"  📄 {f} ({sz} bytes)")
