# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
from datetime import datetime

st.set_page_config(page_title="台股個股分析", page_icon="📊", layout="wide")

st.markdown("""
<style>
.alert-red {background:#4a1a1a;color:#ff6b6b;padding:4px 10px;border-radius:12px;display:inline-block;margin:2px;font-size:13px;}
.alert-yellow {background:#4a3a1a;color:#ffd93d;padding:4px 10px;border-radius:12px;display:inline-block;margin:2px;font-size:13px;}
.alert-green {background:#1a4a2a;color:#6bcf7f;padding:4px 10px;border-radius:12px;display:inline-block;margin:2px;font-size:13px;}
.metric-box {background:#1e1e1e;padding:14px;border-radius:8px;border:1px solid #333;text-align:center;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_dl():
    token = st.secrets["FINMIND_TOKEN"]
    dl = DataLoader()
    dl.login_by_token(api_token=token)
    return dl

dl = get_dl()


# ── 技術指標（純 pandas，零外部依賴）──
def sma(s, n): return s.rolling(n, min_periods=1).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.where(d > 0, 0).ewm(com=n-1, min_periods=n).mean()
    l = (-d.where(d < 0, 0)).ewm(com=n-1, min_periods=n).mean()
    return 100 - 100 / (1 + g / l)

def macd(s, f=12, sl=26, sig=9):
    m = s.ewm(span=f, adjust=False).mean() - s.ewm(span=sl, adjust=False).mean()
    signal = m.ewm(span=sig, adjust=False).mean()
    return m, signal, m - signal

def kd(hi, lo, cl, kp=9, dp=3):
    ll = lo.rolling(kp, min_periods=1).min()
    hh = hi.rolling(kp, min_periods=1).max()
    rsv = 100 * (cl - ll) / (hh - ll).replace(0, np.nan).fillna(50)
    k = rsv.ewm(com=dp-1, adjust=False).mean()
    d = k.ewm(com=dp-1, adjust=False).mean()
    return k, d

def safe(series, idx=-1):
    try:
        v = series.iloc[idx]
        return float(v) if pd.notna(v) else None
    except: return None


# ── 主分析函式 ──
@st.cache_data(ttl=1800, show_spinner=False)
def analyze(stock_id):
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
    is_etf = stock_id.startswith("00") and len(stock_id) >= 5

    # 股價
    df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start, end_date=end)
    if df.empty:
        return None, "無股價資料"

    df = df.rename(columns={"max":"high","min":"low","Trading_Volume":"volume"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 股名
    try:
        info = dl.taiwan_stock_info()
        m = info[info["stock_id"]==stock_id]
        name = m["stock_name"].iloc[0] if not m.empty else stock_id
    except: name = stock_id

    # 技術指標
    df["MA5"]  = sma(df["close"], 5)
    df["MA20"] = sma(df["close"], 20)
    df["MA60"] = sma(df["close"], 60)
    df["RSI"]  = rsi(df["close"])
    df["MACD"], df["MACD_sig"], df["MACD_hist"] = macd(df["close"])
    df["K"], df["D"] = kd(df["high"], df["low"], df["close"])
    df["VMA5"]  = sma(df["volume"], 5)
    df["VRatio"] = df["volume"] / df["VMA5"]
    df["Chg%"]   = df["close"].pct_change() * 100

    lat = df.iloc[-1]
    rsi_v  = safe(df["RSI"])
    k_v    = safe(df["K"])
    d_v    = safe(df["D"])
    ma5_v  = safe(df["MA5"])
    ma20_v = safe(df["MA20"])
    ma60_v = safe(df["MA60"])
    cl_v   = safe(df["close"])
    vr_v   = safe(df["VRatio"])
    chg    = safe(df["Chg%"]) or 0

    # 法人
    i_start = (df["date"].max() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    pivot = pd.DataFrame()
    try:
        inst = dl.taiwan_stock_institutional_investors(
            stock_id=stock_id, start_date=i_start, end_date=end)
        if not inst.empty:
            inst["net"] = inst["buy"] - inst["sell"]
            def cls(n):
                if n in ["Foreign_Investor","Foreign_Dealer_Self"]: return "外資"
                if n == "Investment_Trust": return "投信"
                if n in ["Dealer_self","Dealer_Hedging"]: return "自營商"
                return "其他"
            inst["類別"] = inst["name"].apply(cls)
            p = inst.pivot_table(index="date",columns="類別",values="net",aggfunc="sum").fillna(0)
            for c in ["外資","投信","自營商"]:
                if c not in p.columns: p[c] = 0
            p["合計"] = p["外資"] + p["投信"] + p["自營商"]
            pivot = (p[["外資","投信","自營商","合計"]] / 1000).round().astype(int)
            pivot.index = pd.to_datetime(pivot.index)
            pivot = pivot.sort_index(ascending=False)
    except: pass

    def consec(s):
        if s.empty: return 0, "中性"
        v0 = s.iloc[0]
        if v0 == 0: return 0, "中性"
        dir_ = "買超" if v0 > 0 else "賣超"
        cnt = sum(1 for v in s if (v > 0) == (dir_ == "買超"))
        return cnt, dir_

    if not pivot.empty:
        fd, fdir = consec(pivot["外資"])
        itot = int(pivot["合計"].iloc[0])
        ifor = int(pivot["外資"].iloc[0])
        itru = int(pivot["投信"].iloc[0])
        idal = int(pivot["自營商"].iloc[0])
    else:
        fd, fdir, itot, ifor, itru, idal = 0, "中性", 0, 0, 0, 0

    # 月營收
    yoy = mom = rev = 0
    has_rev = False
    if not is_etf:
        try:
            r_start = (df["date"].max() - pd.Timedelta(days=550)).strftime("%Y-%m-%d")
            rv = dl.taiwan_stock_month_revenue(stock_id=stock_id, start_date=r_start, end_date=end)
            if not rv.empty:
                rv["date"] = pd.to_datetime(rv["date"])
                rv = rv.sort_values("date").reset_index(drop=True)
                rv["MoM"] = rv["revenue"].pct_change(1)*100
                rv["YoY"] = rv["revenue"].pct_change(12)*100
                lr = rv.iloc[-1]
                yoy = float(lr["YoY"]) if pd.notna(lr["YoY"]) else 0
                mom = float(lr["MoM"]) if pd.notna(lr["MoM"]) else 0
                rev = float(lr["revenue"]) / 1e8
                has_rev = True
        except: pass

    # 警示
    alerts = {"red":[], "yellow":[], "green":[]}
    if rsi_v:
        if rsi_v > 80: alerts["red"].append("RSI 嚴重超買")
        elif rsi_v > 70: alerts["yellow"].append("RSI 接近超買")
        elif rsi_v < 30: alerts["green"].append("RSI 超賣可能反彈")
    if k_v and d_v:
        if k_v > 80 and d_v > 80: alerts["red"].append("KD 高檔鈍化")
        elif k_v < 20 and d_v < 20: alerts["green"].append("KD 低檔鈍化")
        if len(df) >= 2:
            pk, pd_ = safe(df["K"],-2), safe(df["D"],-2)
            if pk and pd_:
                if k_v < d_v and pk > pd_: alerts["red"].append("KD 死叉")
                elif k_v > d_v and pk < pd_: alerts["green"].append("KD 黃金交叉")
    if all(v is not None for v in [ma5_v, ma20_v, ma60_v, cl_v]):
        if cl_v > ma5_v > ma20_v > ma60_v: alerts["green"].append("均線多頭排列")
        elif cl_v < ma5_v < ma20_v < ma60_v: alerts["red"].append("均線空頭排列")
    if vr_v:
        if vr_v > 2: alerts["yellow"].append(f"爆量 ({vr_v:.1f}x)")
        elif vr_v < 0.5: alerts["yellow"].append("量縮警示")
    if chg > 0 and itot < 0: alerts["red"].append("籌碼背離")
    elif chg < 0 and itot > 0: alerts["green"].append("法人逆勢買進")
    if fd >= 3:
        (alerts["green"] if fdir=="買超" else alerts["red"]).append(f"外資連{fd}{fdir}")
    if has_rev:
        if yoy > 30: alerts["green"].append(f"營收YoY +{yoy:.0f}%")
        elif yoy > 10: alerts["green"].append(f"營收YoY +{yoy:.0f}%")
        elif yoy < -10: alerts["red"].append(f"營收YoY {yoy:.0f}%")

    # 評分
    ts = 3
    if rsi_v:
        if rsi_v > 80 or (k_v and d_v and k_v > 80 and d_v > 80): ts = 5
        elif rsi_v > 70: ts = 4
        elif cl_v and ma5_v and ma20_v and cl_v > ma5_v > ma20_v: ts = 4
        elif cl_v and ma20_v and cl_v < ma20_v: ts = 2
    cs = 3
    if chg > 0 and itot < 0: cs = 2
    elif fd >= 3 and fdir == "買超": cs = 5
    elif fd >= 3 and fdir == "賣超": cs = 2
    elif itot > 0: cs = 4
    fs = 3
    if has_rev:
        if yoy > 30: fs = 5
        elif yoy > 10: fs = 4
        elif yoy < 0: fs = 2
        elif yoy < -10: fs = 1

    nr = len(alerts["red"])
    ng = len(alerts["green"])
    status = "🔴 過熱" if nr >= 2 else "🟡 觀察" if nr >= 1 else "🟢 健康" if ng >= 2 else "⚪ 中性"

    return {
        "name": name, "id": stock_id, "is_etf": is_etf, "has_rev": has_rev,
        "df": df, "pivot": pivot,
        "close": float(lat["close"]), "chg": chg,
        "vol": int(lat["volume"]/1000),
        "rsi": rsi_v, "k": k_v, "d": d_v,
        "ma5": ma5_v, "ma20": ma20_v, "ma60": ma60_v,
        "macd": safe(df["MACD"]), "macd_sig": safe(df["MACD_sig"]),
        "vr": vr_v,
        "ifor": ifor, "itru": itru, "idal": idal, "itot": itot,
        "fd": fd, "fdir": fdir,
        "yoy": yoy, "mom": mom, "rev": rev,
        "ts": ts, "cs": cs, "fs": fs,
        "status": status,
        "alerts": alerts,
        "rev_df": None,
    }, None


def plot_kline(df, name, sid):
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.44,0.12,0.14,0.15,0.15],
        subplot_titles=("日K線+均線","成交量","RSI(14)","MACD(12,26,9)","KD(9,3,3)"))

    fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color="red", decreasing_line_color="green", name="K線"), row=1, col=1)
    for col, color in [("MA5","yellow"),("MA20","orange"),("MA60","magenta")]:
        fig.add_trace(go.Scatter(x=df["date"], y=df[col], name=col,
            line=dict(color=color, width=1)), row=1, col=1)

    vc = ["red" if c >= o else "green" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], marker_color=vc,
        name="量", showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=df["date"], y=df["RSI"], name="RSI",
        line=dict(color="yellow",width=1.5)), row=3, col=1)
    for y, c in [(70,"red"),(30,"green")]:
        fig.add_hline(y=y, line_dash="dash", line_color=c, row=3, col=1)

    if df["MACD_hist"].notna().any():
        hc = ["red" if v >= 0 else "green" for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df["date"], y=df["MACD_hist"], marker_color=hc,
            name="柱", showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MACD"], name="DIF",
            line=dict(color="cyan",width=1.5)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MACD_sig"], name="DEA",
            line=dict(color="orange",width=1.5)), row=4, col=1)

    fig.add_trace(go.Scatter(x=df["date"], y=df["K"], name="K",
        line=dict(color="cyan",width=1.5)), row=5, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["D"], name="D",
        line=dict(color="magenta",width=1.5)), row=5, col=1)
    for y, c in [(80,"red"),(20,"green")]:
        fig.add_hline(y=y, line_dash="dash", line_color=c, row=5, col=1)

    fig.update_layout(title=f"{name}（{sid}）技術分析", template="plotly_dark",
        height=820, xaxis_rangeslider_visible=False, hovermode="x unified")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat","mon"])])
    return fig


def plot_inst(pivot, df):
    if pivot.empty: return None
    rec = pivot.head(10).sort_index()
    pr  = df[df["date"].isin(pd.to_datetime(rec.index))][["date","close"]].sort_values("date")
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    for col, color in [("外資","#3b82f6"),("投信","#ef4444"),("自營商","#a855f7"),("合計","#22c55e")]:
        fig.add_trace(go.Bar(x=rec.index, y=rec[col], name=col, marker_color=color), secondary_y=False)
    fig.add_trace(go.Scatter(x=pr["date"], y=pr["close"], name="股價",
        line=dict(color="yellow",width=2), marker=dict(size=7), mode="lines+markers"), secondary_y=True)
    fig.update_layout(title="三大法人近10日籌碼動向", template="plotly_dark",
        barmode="group", height=380, hovermode="x unified")
    fig.update_yaxes(title_text="法人買賣超(張)", secondary_y=False)
    fig.update_yaxes(title_text="股價(元)", secondary_y=True)
    return fig


# ══════════════════════════════════════════
#  主畫面
# ══════════════════════════════════════════
st.title("📊 台股個股分析")

# ── 唯一輸入框 ──
col_in, col_btn = st.columns([3, 1])
with col_in:
    sid = st.text_input("", placeholder="輸入股票代號，例如 2330、0050、00981A",
                        label_visibility="collapsed")
with col_btn:
    go_btn = st.button("🔍 開始分析", type="primary", use_container_width=True)

st.divider()

if not (go_btn and sid.strip()):
    st.markdown("""
    ### 👋 使用方式
    1. 在上方輸入框輸入**股票代號**（上市/上櫃/ETF 都可以）
    2. 按「🔍 開始分析」
    3. 查看完整的 **技術面 + 籌碼面 + 基本面** 報告

    ---
    #### 支援的類型
    | 類型 | 範例 |
    |------|------|
    | 上市 | 2330 台積電、2303 聯電、6849 奇鼎 |
    | 上櫃 | 4573 高明鐵 |
    | ETF  | 0050、00981A |
    | 興櫃 | 7731 火星生技（資料可能較少） |
    """)
    st.stop()

sid = sid.strip().upper()

with st.spinner(f"分析 {sid} 中，請稍候..."):
    result, err = analyze(sid)

if err or result is None:
    st.error(f"❌ {err or '分析失敗，請確認股票代號正確'}")
    st.stop()

r = result
df = r["df"]

# ── 標題列 ──
chg_color = "#ff6b6b" if r["chg"] >= 0 else "#6bcf7f"
chg_arrow = "▲" if r["chg"] >= 0 else "▼"
st.markdown(f"""
<h2 style="margin-bottom:4px;">{r['name']} <span style="color:#888;font-size:18px;">{r['id']}</span>
&nbsp;&nbsp;
<span style="color:{chg_color};font-size:24px;">{r['close']:.2f}
&nbsp;{chg_arrow} {abs(r['chg']):.2f}%</span>
&nbsp;&nbsp;
<span style="font-size:16px;color:#888;">{r['status']}</span>
</h2>
<p style="color:#888;font-size:13px;">成交量：{r['vol']:,} 張 &nbsp;|&nbsp; 最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
""", unsafe_allow_html=True)

# ── 警示標籤 ──
alerts_html = ""
for a in r["alerts"]["red"]: alerts_html += f'<span class="alert-red">🔴 {a}</span> '
for a in r["alerts"]["yellow"]: alerts_html += f'<span class="alert-yellow">🟡 {a}</span> '
for a in r["alerts"]["green"]: alerts_html += f'<span class="alert-green">🟢 {a}</span> '
if alerts_html:
    st.markdown(alerts_html, unsafe_allow_html=True)

st.divider()

# ── 指標速覽（一排 6 格）──
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("RSI(14)", f"{r['rsi']:.1f}" if r['rsi'] else "N/A",
    "⚠ 超買" if r['rsi'] and r['rsi']>80 else "🟢 超賣" if r['rsi'] and r['rsi']<30 else "")
c2.metric("K / D", f"{r['k']:.0f} / {r['d']:.0f}" if r['k'] and r['d'] else "N/A")
c3.metric("MACD", f"{r['macd']:.2f}" if r['macd'] else "N/A")
c4.metric("MA5 / MA20", f"{r['ma5']:.1f} / {r['ma20']:.1f}" if r['ma5'] and r['ma20'] else "N/A")
c5.metric("量比", f"{r['vr']:.2f}x" if r['vr'] else "N/A",
    "⚡ 爆量" if r['vr'] and r['vr']>2 else "")
c6.metric("評分 技/籌/基", f"{r['ts']}/5 · {r['cs']}/5 · {r['fs']}/5")

st.divider()

# ── 法人速覽 ──
ca,cb,cc,cd = st.columns(4)
ca.metric("外資(張)", f"{r['ifor']:+,}", f"{r['fd']}日{r['fdir']}")
cb.metric("投信(張)", f"{r['itru']:+,}")
cc.metric("自營商(張)", f"{r['idal']:+,}")
cd.metric("法人合計(張)", f"{r['itot']:+,}")

if r["has_rev"]:
    st.divider()
    ce,cf,cg = st.columns(3)
    ce.metric("最新月營收(億)", f"{r['rev']:.2f}")
    cf.metric("YoY", f"{r['yoy']:+.1f}%",
        "🟢" if r['yoy']>10 else "🔴" if r['yoy']<-10 else "")
    cg.metric("MoM", f"{r['mom']:+.1f}%",
        "🟢" if r['mom']>5 else "🔴" if r['mom']<-5 else "")

st.divider()

# ── 圖表 ──
st.plotly_chart(plot_kline(df, r["name"], r["id"]), use_container_width=True)

if not r["pivot"].empty:
    fig_inst = plot_inst(r["pivot"], df)
    if fig_inst:
        st.plotly_chart(fig_inst, use_container_width=True)

    st.subheader("💼 三大法人近10日明細（張）")
    st.dataframe(r["pivot"].head(10), use_container_width=True)

st.caption(f"⚠️ 本分析僅供參考，不代表投資建議。操作前請依個人資金控管與風險承受度判斷。")
