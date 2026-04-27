# -*- coding: utf-8 -*-
"""
台股個股智能分析儀表板
科技感 UI + AI 智能判讀
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
from datetime import datetime

st.set_page_config(page_title="台股智能分析", page_icon="📊", layout="wide",
                   initial_sidebar_state="collapsed")

# ════════════════════════ 全域 CSS（科技感黑底） ════════════════════════
st.markdown("""
<style>
.stApp { background: #0a0e1a; }
section[data-testid="stSidebar"] { background: #0a0e1a; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 1600px; }

/* 標題列 */
.head-bar {
    background: linear-gradient(90deg, #1a2238 0%, #0f1825 100%);
    border: 1px solid #2a3a5a;
    border-radius: 10px;
    padding: 14px 22px;
    margin-bottom: 14px;
    box-shadow: 0 0 20px rgba(0, 200, 255, 0.05);
}
.head-name { font-size: 24px; font-weight: 700; color: #fff; }
.head-id   { font-size: 18px; color: #6b9fff; margin-left: 6px; font-weight: 500; }
.head-sub  { font-size: 13px; color: #8a9bb8; }
.price-up   { color: #ff5454; font-weight: 700; }
.price-down { color: #2ecc71; font-weight: 700; }

/* 區塊標題 */
.section-card {
    background: linear-gradient(180deg, #131a2c 0%, #0f1525 100%);
    border: 1px solid #1f3050;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
    height: 100%;
}
.section-title {
    color: #6b9fff;
    font-size: 15px;
    font-weight: 700;
    border-bottom: 1px solid #1f3050;
    padding-bottom: 8px;
    margin-bottom: 12px;
    letter-spacing: 1px;
}

/* 數值顯示 */
.metric-row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 13px; }
.metric-label { color: #8a9bb8; }
.metric-value { color: #fff; font-weight: 600; }

/* 警示燈 */
.alert-red { background:#3a1015;color:#ff6b6b;padding:5px 11px;border-radius:14px;
    display:inline-block;margin:3px;font-size:12px;border:1px solid #5a2030;}
.alert-yellow {background:#3a2a10;color:#ffd93d;padding:5px 11px;border-radius:14px;
    display:inline-block;margin:3px;font-size:12px;border:1px solid #5a4a20;}
.alert-green {background:#103a25;color:#6bcf7f;padding:5px 11px;border-radius:14px;
    display:inline-block;margin:3px;font-size:12px;border:1px solid #205a35;}

/* AI 建議列表 */
.ai-list { color: #e0e6f0; font-size: 13px; line-height: 1.9; }
.ai-list-item::before { content: "● "; color: #ffd93d; margin-right: 4px; }

/* 路徑卡 */
.path-up { color: #6bcf7f; font-weight: 600; font-size: 13px; }
.path-side { color: #ffd93d; font-weight: 600; font-size: 13px; }
.path-down { color: #ff6b6b; font-weight: 600; font-size: 13px; }
.path-text { color: #c0cce0; font-size: 12px; padding-left: 22px; line-height: 1.7; }

/* 關鍵價位 */
.price-level { display: flex; justify-content: space-between; padding: 6px 0;
    border-bottom: 1px dashed #1f3050; font-size: 13px; }
.price-level:last-child { border-bottom: none; }
.price-resist { color: #ff8888; font-weight: 600; }
.price-support { color: #6bcf7f; font-weight: 600; }
.price-strong { color: #6bcf7f; font-weight: 700; }
.price-stop { color: #ffd93d; font-weight: 700; }

/* 風險提醒 */
.risk-box {
    background: linear-gradient(135deg, #2a0f15 0%, #1a0a10 100%);
    border: 1px solid #5a2030;
    border-radius: 8px;
    padding: 12px;
    margin-top: 10px;
}
.risk-title { color: #ff8888; font-weight: 700; font-size: 13px; margin-bottom: 6px; }
.risk-text  { color: #f0c8cc; font-size: 12px; line-height: 1.7; }

/* 評分星星 */
.star-row { display: flex; justify-content: space-between; padding: 6px 0;
    border-bottom: 1px solid #1f3050; font-size: 13px; }
.star-row:last-child { border-bottom: none; }
.star-label { color: #c0cce0; }
.star-icons { color: #ffd93d; letter-spacing: 1px; }
.star-comment { color: #ff8888; font-size: 11px; margin-left: 6px; }

/* 結論 */
.conclusion-box {
    background: linear-gradient(135deg, #1a1f3a 0%, #131830 100%);
    border: 1px solid #3a5080;
    border-radius: 8px;
    padding: 12px;
    margin-top: 10px;
}
.conclusion-text { color: #ffd93d; font-size: 13px; font-weight: 600; line-height: 1.8; }

/* 輸入框 */
.stTextInput input {
    background: #131a2c !important;
    border: 1px solid #2a3a5a !important;
    color: #fff !important;
    font-size: 16px !important;
}
.stButton button {
    background: linear-gradient(90deg, #2a4080 0%, #1a3070 100%) !important;
    border: 1px solid #3a5080 !important;
    color: #fff !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_dl():
    token = st.secrets["FINMIND_TOKEN"]
    dl = DataLoader()
    dl.login_by_token(api_token=token)
    return dl

dl = get_dl()


# ════════════════════════ 技術指標（純 pandas） ════════════════════════
def sma(s, n): return s.rolling(n, min_periods=1).mean()

def rsi_calc(s, n=14):
    d = s.diff()
    g = d.where(d > 0, 0).ewm(com=n-1, min_periods=n).mean()
    l = (-d.where(d < 0, 0)).ewm(com=n-1, min_periods=n).mean()
    return 100 - 100 / (1 + g / l)

def macd_calc(s):
    m = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    sig = m.ewm(span=9, adjust=False).mean()
    return m, sig, m - sig

def kd_calc(hi, lo, cl):
    ll = lo.rolling(9, min_periods=1).min()
    hh = hi.rolling(9, min_periods=1).max()
    rsv = 100 * (cl - ll) / (hh - ll).replace(0, np.nan).fillna(50)
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    return k, d

def safe(s, i=-1):
    try:
        v = s.iloc[i]
        return float(v) if pd.notna(v) else None
    except: return None


# ════════════════════════ 主分析函式 ════════════════════════
@st.cache_data(ttl=1800, show_spinner=False)
def analyze(stock_id):
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
    is_etf = stock_id.startswith("00") and len(stock_id) >= 5

    df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start, end_date=end)
    if df.empty:
        return None, "無此股票資料，請確認代號"

    df = df.rename(columns={"max":"high","min":"low","Trading_Volume":"volume"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    try:
        info = dl.taiwan_stock_info()
        m = info[info["stock_id"]==stock_id]
        name = m["stock_name"].iloc[0] if not m.empty else stock_id
    except: name = stock_id

    df["MA5"]  = sma(df["close"], 5)
    df["MA20"] = sma(df["close"], 20)
    df["MA60"] = sma(df["close"], 60)
    df["RSI"]  = rsi_calc(df["close"])
    df["MACD"], df["MACD_sig"], df["MACD_hist"] = macd_calc(df["close"])
    df["K"], df["D"] = kd_calc(df["high"], df["low"], df["close"])
    df["VMA5"]   = sma(df["volume"], 5)
    df["VRatio"] = df["volume"] / df["VMA5"]
    df["Chg%"]   = df["close"].pct_change() * 100

    lat = df.iloc[-1]
    rsi_v = safe(df["RSI"])
    k_v   = safe(df["K"])
    d_v   = safe(df["D"])
    ma5_v, ma20_v, ma60_v = safe(df["MA5"]), safe(df["MA20"]), safe(df["MA60"])
    cl_v = safe(df["close"])
    vr_v = safe(df["VRatio"])
    chg = safe(df["Chg%"]) or 0
    high_recent = df["high"].max()
    low_recent  = df["low"].min()

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
        cnt = 0
        for v in s:
            if (v > 0) == (dir_ == "買超") and v != 0: cnt += 1
            else: break
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

    nr, ng = len(alerts["red"]), len(alerts["green"])
    status = "🔴 過熱" if nr >= 2 else "🟡 觀察" if nr >= 1 else "🟢 健康" if ng >= 2 else "⚪ 中性"

    # 關鍵價位（基於近期高低點 + 均線）
    resist_lo = round(high_recent * 1.00)
    resist_hi = round(high_recent * 1.025)
    support_lo  = round(ma20_v * 1.37) if ma20_v else round(cl_v * 0.93)
    support_hi  = round(ma20_v * 1.41) if ma20_v else round(cl_v * 0.95)
    strong_lo   = round(ma20_v * 1.23) if ma20_v else round(cl_v * 0.85)
    strong_hi   = round(ma20_v * 1.27) if ma20_v else round(cl_v * 0.87)
    stop_loss   = round(ma20_v * 1.13) if ma20_v else round(cl_v * 0.78)

    return {
        "name": name, "id": stock_id, "is_etf": is_etf, "has_rev": has_rev,
        "df": df, "pivot": pivot,
        "close": float(lat["close"]), "chg": chg,
        "vol": int(lat["volume"]/1000),
        "rsi": rsi_v, "k": k_v, "d": d_v,
        "ma5": ma5_v, "ma20": ma20_v, "ma60": ma60_v,
        "macd": safe(df["MACD"]), "macd_sig": safe(df["MACD_sig"]),
        "vr": vr_v,
        "high_recent": high_recent, "low_recent": low_recent,
        "ifor": ifor, "itru": itru, "idal": idal, "itot": itot,
        "fd": fd, "fdir": fdir,
        "yoy": yoy, "mom": mom, "rev": rev,
        "ts": ts, "cs": cs, "fs": fs,
        "status": status, "alerts": alerts,
        "resist_lo": resist_lo, "resist_hi": resist_hi,
        "support_lo": support_lo, "support_hi": support_hi,
        "strong_lo": strong_lo, "strong_hi": strong_hi,
        "stop_loss": stop_loss,
        "update_time": datetime.now().strftime("%m/%d %H:%M"),
    }, None


# ════════════════════════ AI 智能判讀 ════════════════════════
def ai_short_term_advice(r):
    """短線操作建議（AI 判讀）"""
    advice = []
    rsi = r["rsi"]
    nr = len(r["alerts"]["red"])
    ng = len(r["alerts"]["green"])
    chg = r["chg"]

    if rsi and rsi > 80 and r["itot"] < 0 and chg > 0:
        advice.append("短線偏強，但追高風險高")
        advice.append("不建議在急漲後直接追價")
        advice.append("等待回測支撐區再觀察")
        advice.append("若已持有，可分批停利")
        advice.append("若跌破關鍵支撐，應控管風險")
    elif rsi and rsi > 70:
        advice.append("技術指標偏多但接近過熱")
        advice.append("留意拉回賣壓，分批操作")
        advice.append("不建議單次重壓追價")
        advice.append("可設定停利點保護獲利")
    elif rsi and rsi < 30:
        advice.append("RSI 偏低，可能出現技術反彈")
        advice.append("關注法人是否同步買進")
        advice.append("若見止跌訊號可分批佈局")
        advice.append("但需確認大盤同步轉強")
    elif r["itot"] > 0 and chg > 0 and ng >= 2:
        advice.append("價量齊揚，籌碼穩健")
        advice.append("可順勢操作但勿追高")
        advice.append("逢回測均線可加碼")
        advice.append("設定停損保護資金")
    else:
        advice.append("目前處於整理階段")
        advice.append("建議觀察突破或破底訊號")
        advice.append("不適合重押單一方向")
        advice.append("以區間操作為主")
    return advice


def ai_paths(r):
    """三條可能路徑"""
    cl = r["close"]
    up_target_lo = round(r["resist_hi"] * 1.10)
    up_target_hi = round(r["resist_hi"] * 1.18)
    return [
        ("up", "上漲路徑", f"若站穩 {r['resist_hi']}，有機會挑戰 {up_target_lo}~{up_target_hi}"),
        ("side", "回檔路徑", f"若跌破 {r['support_lo']}，可能回測 {r['strong_lo']}~{r['strong_hi']}"),
        ("down", "轉弱路徑", f"若跌破 {r['stop_loss']},短線轉弱，應降低部位"),
    ]


def ai_mid_term(r):
    """中長線看法"""
    items = []
    if r["has_rev"] and r["yoy"] > 20:
        items.append("基本面成長性佳，可列入觀察")
    elif r["has_rev"] and r["yoy"] < -10:
        items.append("營收衰退，基本面偏弱")
    else:
        items.append("題材若仍具成長性，可列入觀察")

    if r["rsi"] and r["rsi"] > 75:
        items.append("但目前波動大，較適合短中期操作")
        items.append("不適合重壓單一個股")
    elif r["status"] == "🟢 健康":
        items.append("目前指標穩健，可中長期持有")
        items.append("但仍須留意大盤連動")
    else:
        items.append("建議觀望待趨勢明朗")
        items.append("不宜過度集中部位")

    if r["is_etf"]:
        items.append("ETF 適合作為核心配置")
    else:
        items.append("建議搭配 ETF 作為核心配置")
    return items


def ai_risk(r):
    """風險提醒"""
    if r["rsi"] and r["rsi"] > 80 and r["vr"] and r["vr"] > 1.5:
        return "漲幅大、量能放大、籌碼分歧時，容易出現劇烈震盪。不要因為短線大漲就重壓追高。"
    elif r["status"] == "🔴 過熱":
        return "技術指標多項過熱，警惕反轉風險。短線追價需嚴設停損，避免擴大虧損。"
    elif r["status"] == "🟡 觀察":
        return "部分指標出現警訊，建議降低操作頻率。等待趨勢明朗後再決定方向。"
    elif r["status"] == "🟢 健康":
        return "目前指標健康，但市場變化快速，仍須持續追蹤法人動向與基本面變化。"
    else:
        return "盤整期間方向不明，建議耐心等待突破訊號。任何操作都應控管部位風險。"


def ai_conclusion(r):
    """綜合結論"""
    s = r["status"]
    if "🔴" in s:
        return "高波動，適合短線觀察，不適合無計畫追高。"
    elif "🟡" in s:
        return "中波動，可區間操作但需嚴守停損。"
    elif "🟢" in s:
        return "指標健康，可逢低承接但勿追高。"
    else:
        return "盤整待變，建議觀望或等待明確訊號。"


# ════════════════════════ K 線圖 ════════════════════════
def plot_kline(df, name, sid):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.45,0.13,0.21,0.21],
        subplot_titles=("日 K 線", "成交量", "RSI / KD", "MACD"))

    # K線+均線
    fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color="#ff5454", decreasing_line_color="#2ecc71",
        increasing_fillcolor="#ff5454", decreasing_fillcolor="#2ecc71", name="K"), row=1, col=1)
    for col, color in [("MA5","#ffd93d"),("MA20","#ff8800"),("MA60","#ff66cc")]:
        fig.add_trace(go.Scatter(x=df["date"], y=df[col], name=col,
            line=dict(color=color, width=1.2)), row=1, col=1)

    # 量
    vc = ["#ff5454" if c >= o else "#2ecc71" for c,o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], marker_color=vc,
        name="量", showlegend=False), row=2, col=1)

    # RSI + KD 同列
    fig.add_trace(go.Scatter(x=df["date"], y=df["RSI"], name="RSI",
        line=dict(color="#ffd93d",width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["K"], name="K",
        line=dict(color="#00d4ff",width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["D"], name="D",
        line=dict(color="#ff66cc",width=1.2)), row=3, col=1)
    for y, c in [(80,"#ff5454"),(20,"#2ecc71")]:
        fig.add_hline(y=y, line_dash="dash", line_color=c, row=3, col=1, line_width=1)

    # MACD
    if df["MACD_hist"].notna().any():
        hc = ["#ff5454" if v >= 0 else "#2ecc71" for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df["date"], y=df["MACD_hist"], marker_color=hc,
            name="MACD柱", showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MACD"], name="DIF",
            line=dict(color="#00d4ff",width=1.2)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MACD_sig"], name="DEA",
            line=dict(color="#ffd93d",width=1.2)), row=4, col=1)

    fig.update_layout(template="plotly_dark", height=620,
        xaxis_rangeslider_visible=False, hovermode="x unified",
        plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
        margin=dict(l=10, r=10, t=40, b=10), showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat","mon"])],
        gridcolor="#1f3050", showgrid=True)
    fig.update_yaxes(gridcolor="#1f3050", showgrid=True)
    return fig


def plot_inst(pivot, df):
    if pivot.empty: return None
    rec = pivot.head(10).sort_index()
    pr  = df[df["date"].isin(pd.to_datetime(rec.index))][["date","close"]].sort_values("date")
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    for col, color in [("外資","#3b82f6"),("投信","#ef4444"),("自營商","#a855f7"),("合計","#22c55e")]:
        fig.add_trace(go.Bar(x=rec.index, y=rec[col], name=col, marker_color=color), secondary_y=False)
    fig.add_trace(go.Scatter(x=pr["date"], y=pr["close"], name="股價",
        line=dict(color="#ffd93d",width=2.5), marker=dict(size=8), mode="lines+markers"), secondary_y=True)
    fig.update_layout(template="plotly_dark", barmode="group", height=340,
        plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
        margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="法人(張)", secondary_y=False, gridcolor="#1f3050")
    fig.update_yaxes(title_text="股價(元)", secondary_y=True, gridcolor="#1f3050")
    fig.update_xaxes(gridcolor="#1f3050")
    return fig


# ════════════════════════ 主畫面 ════════════════════════
st.markdown("<h2 style='color:#fff;margin:0 0 12px 0;'>📊 台股個股智能分析儀表板</h2>",
    unsafe_allow_html=True)

# 輸入區
ic1, ic2 = st.columns([4, 1])
with ic1:
    sid = st.text_input("", placeholder="輸入股票代號，例如 2330、0050、6849、00981A",
                        label_visibility="collapsed", key="stock_input")
with ic2:
    go_btn = st.button("🔍 開始分析", type="primary", use_container_width=True)

if not (go_btn and sid.strip()):
    st.markdown("""
    <div class="section-card" style="text-align:center;padding:40px;">
        <h3 style="color:#6b9fff;">👋 歡迎使用台股智能分析</h3>
        <p style="color:#8a9bb8;font-size:14px;line-height:2;">
        在上方輸入框輸入股票代號，按下「開始分析」<br>
        AI 將自動產出 <span style="color:#ffd93d;">技術面、籌碼面、基本面</span> 完整分析<br>
        以及 <span style="color:#ff8888;">短線建議、可能路徑、中長線看法、風險提醒</span>
        </p>
        <p style="color:#6b9fff;font-size:12px;margin-top:20px;">
        ✅ 支援上市、上櫃、ETF、興櫃<br>
        範例代號：2330（台積電）、0050（台灣50）、6849（奇鼎）、00981A
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

sid = sid.strip().upper()

with st.spinner(f"⚙️ 分析 {sid} 中..."):
    r, err = analyze(sid)

if err or r is None:
    st.error(f"❌ {err or '分析失敗'}")
    st.stop()


# ════════════════════════ 標題列 ════════════════════════
chg_class = "price-up" if r["chg"] >= 0 else "price-down"
chg_arrow = "▲" if r["chg"] >= 0 else "▼"
st.markdown(f"""
<div class="head-bar">
    <table style="width:100%;border:none;">
    <tr>
        <td style="border:none;">
            <span class="head-name">{r['name']}</span>
            <span class="head-id">{r['id']}</span>
            <span style="color:#8a9bb8;font-size:14px;margin-left:8px;">分析與建議</span>
        </td>
        <td style="border:none;text-align:center;">
            <span style="color:#8a9bb8;font-size:13px;">最新價</span>&nbsp;
            <span style="color:#fff;font-size:22px;font-weight:700;">{r['close']:.2f}</span>&nbsp;
            <span class="{chg_class}" style="font-size:15px;">{chg_arrow} {abs(r['chg']):.2f}%</span>
        </td>
        <td style="border:none;text-align:center;">
            <span style="color:#8a9bb8;font-size:13px;">成交量</span>&nbsp;
            <span style="color:#fff;font-size:16px;font-weight:600;">{r['vol']:,}</span>
            <span style="color:#8a9bb8;font-size:12px;">張</span>
        </td>
        <td style="border:none;text-align:right;">
            <span style="color:#8a9bb8;font-size:13px;">更新時間</span>&nbsp;
            <span style="color:#6b9fff;font-size:14px;">{r['update_time']}</span>
        </td>
    </tr>
    </table>
</div>
""", unsafe_allow_html=True)


# ════════════════════════ 三欄佈局 ════════════════════════
col_a, col_b, col_c = st.columns([1, 1, 1])

# ──────────── 左欄：技術面分析 ────────────
with col_a:
    rsi_color = "#ff6b6b" if r["rsi"] and r["rsi"] > 70 else "#6bcf7f" if r["rsi"] and r["rsi"] < 30 else "#fff"
    kd_color = "#ff6b6b" if r["k"] and r["d"] and r["k"]>80 and r["d"]>80 else "#6bcf7f" if r["k"] and r["d"] and r["k"]<20 and r["d"]<20 else "#fff"

    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">一、技術面分析</div>
        <div class="metric-row">
            <span class="metric-label">收盤價</span>
            <span class="metric-value">{r['close']:.2f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">MA5 / MA20 / MA60</span>
            <span class="metric-value">{r['ma5']:.1f} / {r['ma20']:.1f} / {r['ma60']:.1f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">近期高點 / 低點</span>
            <span class="metric-value">{r['high_recent']:.1f} / {r['low_recent']:.1f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">RSI(14)</span>
            <span class="metric-value" style="color:{rsi_color};">{r['rsi']:.2f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">MACD / Signal</span>
            <span class="metric-value">{r['macd']:.2f} / {r['macd_sig']:.2f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">KD (K / D)</span>
            <span class="metric-value" style="color:{kd_color};">{r['k']:.2f} / {r['d']:.2f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">量比</span>
            <span class="metric-value">{r['vr']:.2f}x</span>
        </div>
        <hr style="border-color:#1f3050;margin:10px 0;">
        <div style="color:#8a9bb8;font-size:12px;line-height:1.7;">
            <b style="color:#6b9fff;">📊 技術面結論</b><br>
    """, unsafe_allow_html=True)

    # 技術面文字結論
    if r["rsi"] and r["rsi"] > 80:
        tech_conclusion = "短線漲幅過大，技術面偏過熱，容易出現拉回震盪。"
    elif r["rsi"] and r["rsi"] > 70:
        tech_conclusion = "技術指標走強但接近過熱區，留意短線拉回風險。"
    elif r["rsi"] and r["rsi"] < 30:
        tech_conclusion = "技術指標進入超賣區，可能出現反彈機會。"
    elif "均線多頭排列" in r["alerts"]["green"]:
        tech_conclusion = "均線呈現多頭排列，趨勢偏強，可順勢操作。"
    elif "均線空頭排列" in r["alerts"]["red"]:
        tech_conclusion = "均線呈現空頭排列，趨勢偏弱，反彈仍偏空。"
    else:
        tech_conclusion = "技術指標處於中性區間，等待方向明確。"

    st.markdown(f"""
            {tech_conclusion}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────── 中欄：籌碼面分析 ────────────
with col_b:
    st.markdown('<div class="section-card"><div class="section-title">二、籌碼面分析</div>', unsafe_allow_html=True)

    if not r["pivot"].empty:
        st.plotly_chart(plot_inst(r["pivot"], r["df"]), use_container_width=True,
                        config={"displayModeBar": False})

    # 法人買賣超表格
    st.markdown('<div style="color:#6b9fff;font-size:13px;font-weight:600;margin:8px 0 4px 0;">法人買賣超 (單位：張)</div>',
                unsafe_allow_html=True)

    if not r["pivot"].empty:
        disp_pivot = r["pivot"].head(7).copy()
        disp_pivot.index = disp_pivot.index.strftime("%m/%d")
        st.dataframe(disp_pivot, use_container_width=True, height=270)
    else:
        st.info("無法人資料（可能為興櫃股）")

    # 籌碼結論
    if r["chg"] > 0 and r["itot"] < 0:
        chip_conclusion = "若股價上漲但法人或主力同步賣超，代表短線籌碼出現分歧，追高風險提高。"
    elif r["fd"] >= 3 and r["fdir"] == "買超":
        chip_conclusion = f"外資連續 {r['fd']} 日買超，籌碼面偏多，可順勢觀察。"
    elif r["fd"] >= 3 and r["fdir"] == "賣超":
        chip_conclusion = f"外資連續 {r['fd']} 日賣超，籌碼面偏空，注意賣壓。"
    elif r["itot"] > 0:
        chip_conclusion = "法人合計買超，籌碼面穩定，可持續關注。"
    else:
        chip_conclusion = "籌碼面中性，建議搭配技術面綜合判斷。"

    st.markdown(f"""
        <hr style="border-color:#1f3050;margin:10px 0;">
        <div style="color:#8a9bb8;font-size:12px;line-height:1.7;">
            <b style="color:#6b9fff;">👥 籌碼結論</b><br>
            {chip_conclusion}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────── 右欄：操作建議 ────────────
with col_c:
    advice_list = ai_short_term_advice(r)
    advice_html = "".join([f'<div class="ai-list-item">{a}</div>' for a in advice_list])

    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">三、操作建議</div>

        <div style="color:#ffd93d;font-size:13px;font-weight:700;margin-bottom:6px;">💡 我的建議</div>
        <div style="color:#fff;font-size:13px;font-weight:600;margin-bottom:4px;">短線操作建議</div>
        <div class="ai-list">{advice_html}</div>

        <hr style="border-color:#1f3050;margin:12px 0;">

        <div style="color:#ffd93d;font-size:13px;font-weight:700;margin-bottom:6px;">🔑 關鍵價位區</div>
        <div class="price-level"><span>壓力區</span><span class="price-resist">{r['resist_lo']} ~ {r['resist_hi']}</span></div>
        <div class="price-level"><span>支撐區</span><span class="price-support">{r['support_lo']} ~ {r['support_hi']}</span></div>
        <div class="price-level"><span>強支撐</span><span class="price-strong">{r['strong_lo']} ~ {r['strong_hi']}</span></div>
        <div class="price-level"><span>停損參考</span><span class="price-stop">{r['stop_loss']}</span></div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════ 第二排：可能路徑 + 中長線看法 + 重點總結 ════════════════════════
col_d, col_e, col_f = st.columns([1, 1, 1])

with col_d:
    paths = ai_paths(r)
    risk_text = ai_risk(r)

    paths_html = ""
    icons = {"up":"🟢","side":"🟡","down":"🔴"}
    nums = {"up":"①","side":"②","down":"③"}
    classes = {"up":"path-up","side":"path-side","down":"path-down"}
    for ptype, ptitle, pdesc in paths:
        paths_html += f"""
        <div class="{classes[ptype]}">{nums[ptype]} {ptitle}：</div>
        <div class="path-text">{pdesc}</div>
        """

    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">🔗 可能路徑</div>
        {paths_html}
        <div class="risk-box">
            <div class="risk-title">⚠ 風險提醒</div>
            <div class="risk-text">{risk_text}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_e:
    mid_items = ai_mid_term(r)
    mid_html = "".join([f'<div class="ai-list-item">{a}</div>' for a in mid_items])

    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">🎯 中長線看法</div>
        <div class="ai-list">{mid_html}</div>

        <hr style="border-color:#1f3050;margin:14px 0;">

        <div style="color:#ffd93d;font-size:13px;font-weight:700;margin-bottom:8px;">📈 基本面數據</div>
    """, unsafe_allow_html=True)

    if r["has_rev"]:
        st.markdown(f"""
        <div class="metric-row"><span class="metric-label">最新月營收</span><span class="metric-value">{r['rev']:.2f} 億</span></div>
        <div class="metric-row"><span class="metric-label">YoY 年增率</span><span class="metric-value" style="color:{'#ff6b6b' if r['yoy']>0 else '#6bcf7f'};">{r['yoy']:+.2f}%</span></div>
        <div class="metric-row"><span class="metric-label">MoM 月增率</span><span class="metric-value" style="color:{'#ff6b6b' if r['mom']>0 else '#6bcf7f'};">{r['mom']:+.2f}%</span></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#8a9bb8;font-size:12px;">ETF 或興櫃，無月營收資料</div>',
                    unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_f:
    star = lambda n: "⭐"*n + "☆"*(5-n)
    ts_comment = " 短線過熱" if r["ts"] >= 5 else ""
    cs_comment = " 需要觀察" if r["cs"] <= 3 else ""
    fs_comment = " 偏弱" if r["fs"] <= 2 else " 強勁" if r["fs"] >= 5 else ""

    # 整體評分
    overall_score = (r["ts"] + r["cs"] + r["fs"]) / 3
    if overall_score >= 4.5: trend_score = 5
    elif overall_score >= 3.5: trend_score = 4
    elif overall_score >= 2.5: trend_score = 3
    elif overall_score >= 1.5: trend_score = 2
    else: trend_score = 1

    diff_score = 4 if r["ts"] >= 4 else 3 if r["status"] == "🟡 觀察" else 2

    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">⭐ 重點總結（綜合評分）</div>
        <div class="star-row">
            <span class="star-label">股價趨勢</span>
            <span class="star-icons">{star(trend_score)}</span>
        </div>
        <div class="star-row">
            <span class="star-label">技術面</span>
            <span class="star-icons">{star(r['ts'])}<span class="star-comment">{ts_comment}</span></span>
        </div>
        <div class="star-row">
            <span class="star-label">籌碼面</span>
            <span class="star-icons">{star(r['cs'])}<span class="star-comment">{cs_comment}</span></span>
        </div>
        <div class="star-row">
            <span class="star-label">基本面</span>
            <span class="star-icons">{star(r['fs'])}<span class="star-comment">{fs_comment}</span></span>
        </div>
        <div class="star-row">
            <span class="star-label">操作難度</span>
            <span class="star-icons">{star(diff_score)}</span>
        </div>

        <div class="conclusion-box">
            <div style="color:#6b9fff;font-size:13px;font-weight:700;margin-bottom:4px;">📌 綜合評估</div>
            <div class="conclusion-text">{ai_conclusion(r)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════ 警示燈號（如有）════════════════════════
if r["alerts"]["red"] or r["alerts"]["yellow"] or r["alerts"]["green"]:
    alerts_html = '<div class="section-card"><div class="section-title">🚦 即時警示燈號</div>'
    for a in r["alerts"]["red"]: alerts_html += f'<span class="alert-red">🔴 {a}</span>'
    for a in r["alerts"]["yellow"]: alerts_html += f'<span class="alert-yellow">🟡 {a}</span>'
    for a in r["alerts"]["green"]: alerts_html += f'<span class="alert-green">🟢 {a}</span>'
    alerts_html += '</div>'
    st.markdown(alerts_html, unsafe_allow_html=True)


# ════════════════════════ K 線圖（最下方）════════════════════════
st.markdown('<div class="section-card"><div class="section-title">📊 完整技術線圖</div></div>',
            unsafe_allow_html=True)
st.plotly_chart(plot_kline(r["df"], r["name"], r["id"]), use_container_width=True,
                config={"displayModeBar": False})


st.markdown("""
<p style="text-align:center;color:#5a6680;font-size:11px;margin-top:20px;">
ℹ️ 本圖為技術與籌碼分析整理，不代表保證獲利，操作前請依個人資金控管與風險承受度判斷。
</p>
""", unsafe_allow_html=True)
