# -*- coding: utf-8 -*-
"""
台股觀察清單分析網站
作者: Hodichen
功能: 自動抓取觀察股票的技術面、籌碼面、基本面數據，
      並提供綜合警示與評分系統
"""

import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from FinMind.data import DataLoader
import json
import time
import io
from datetime import datetime, timedelta

# ============================================
# 頁面設定
# ============================================
st.set_page_config(
    page_title="台股觀察清單",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂 CSS 樣式
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stMetric { background: #1e1e1e; padding: 10px; border-radius: 8px; }
    .alert-red {
        background: #4a1a1a; color: #ff6b6b;
        padding: 4px 10px; border-radius: 12px;
        display: inline-block; margin: 2px;
        font-size: 12px; font-weight: 500;
    }
    .alert-yellow {
        background: #4a3a1a; color: #ffd93d;
        padding: 4px 10px; border-radius: 12px;
        display: inline-block; margin: 2px;
        font-size: 12px; font-weight: 500;
    }
    .alert-green {
        background: #1a4a2a; color: #6bcf7f;
        padding: 4px 10px; border-radius: 12px;
        display: inline-block; margin: 2px;
        font-size: 12px; font-weight: 500;
    }
    .status-card {
        background: #1e1e1e; padding: 16px;
        border-radius: 8px; border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 載入 FinMind Token (從 Streamlit Secrets)
# ============================================
@st.cache_resource
def get_data_loader():
    """初始化 FinMind 連線"""
    try:
        token = st.secrets["FINMIND_TOKEN"]
        dl = DataLoader()
        dl.login_by_token(api_token=token)
        return dl
    except Exception as e:
        st.error(f"❌ FinMind 登入失敗，請檢查 Token 設定：{e}")
        st.stop()

dl = get_data_loader()

# ============================================
# 觀察清單管理
# ============================================
WATCHLIST_FILE = "watchlist.json"

def load_watchlist():
    """載入觀察清單"""
    try:
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "stocks": [],
            "settings": {
                "default_lookback_days": 180,
                "alert_thresholds": {
                    "rsi_high": 80, "rsi_low": 30,
                    "kd_high": 80, "kd_low": 20
                }
            }
        }

# ============================================
# 核心分析函式 (與 Colab 版本一致)
# ============================================
@st.cache_data(ttl=3600)
def analyze_stock(stock_id, lookback_days=180):
    """
    分析單一台股，回傳完整報告字典
    支援上市、上櫃、興櫃、ETF
    """
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    # 判斷是否為 ETF
    is_etf = stock_id.startswith("00") and len(stock_id) >= 5

    try:
        # --- 1. 抓股價 ---
        df = dl.taiwan_stock_daily(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            return {"股票代號": stock_id, "錯誤": "無股價資料"}

        df = df.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # --- 2. 抓股票名稱 ---
        try:
            info = dl.taiwan_stock_info()
            match = info[info["stock_id"] == stock_id]
            stock_name = match["stock_name"].iloc[0] if not match.empty else stock_id
        except Exception:
            stock_name = stock_id

        # --- 3. 計算技術指標 ---
        df["MA5"] = ta.sma(df["close"], length=5)
        df["MA20"] = ta.sma(df["close"], length=20)
        df["MA60"] = ta.sma(df["close"], length=60)
        df["RSI14"] = ta.rsi(df["close"], length=14)

        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            df["MACD"] = macd["MACD_12_26_9"]
            df["MACD_signal"] = macd["MACDs_12_26_9"]
            df["MACD_hist"] = macd["MACDh_12_26_9"]
        else:
            df["MACD"] = df["MACD_signal"] = df["MACD_hist"] = None

        stoch = ta.stoch(df["high"], df["low"], df["close"], k=9, d=3, smooth_k=3)
        if stoch is not None:
            df["K"] = stoch["STOCHk_9_3_3"]
            df["D"] = stoch["STOCHd_9_3_3"]
        else:
            df["K"] = df["D"] = None

        df["Volume_MA5"] = ta.sma(df["volume"], length=5)
        df["Volume_Ratio"] = df["volume"] / df["Volume_MA5"]
        df["Change_Pct"] = df["close"].pct_change() * 100

        latest = df.iloc[-1]

        # --- 4. 抓法人籌碼 ---
        inst_start = (df["date"].max() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
        inst_end = df["date"].max().strftime("%Y-%m-%d")

        pivot = pd.DataFrame()
        try:
            inst = dl.taiwan_stock_institutional_investors(
                stock_id=stock_id, start_date=inst_start, end_date=inst_end
            )
            if not inst.empty:
                inst["net"] = inst["buy"] - inst["sell"]

                def classify(name):
                    if name in ["Foreign_Investor", "Foreign_Dealer_Self"]:
                        return "外資"
                    elif name == "Investment_Trust":
                        return "投信"
                    elif name in ["Dealer_self", "Dealer_Hedging"]:
                        return "自營商"
                    return "其他"

                inst["類別"] = inst["name"].apply(classify)
                pivot = inst.pivot_table(
                    index="date", columns="類別", values="net", aggfunc="sum"
                ).fillna(0)

                for col in ["外資", "投信", "自營商"]:
                    if col not in pivot.columns:
                        pivot[col] = 0

                pivot["合計"] = pivot["外資"] + pivot["投信"] + pivot["自營商"]
                pivot = pivot[["外資", "投信", "自營商", "合計"]]
                pivot = (pivot / 1000).round().astype(int)
                pivot.index = pd.to_datetime(pivot.index)
                pivot = pivot.sort_index(ascending=False)
        except Exception:
            pass

        # 連續買賣超
        def consecutive_count(series):
            if len(series) == 0:
                return 0, "中性"
            latest_v = series.iloc[0]
            if latest_v == 0:
                return 0, "中性"
            direction = "買超" if latest_v > 0 else "賣超"
            count = 0
            for v in series:
                if (v > 0 and direction == "買超") or (v < 0 and direction == "賣超"):
                    count += 1
                else:
                    break
            return count, direction

        if not pivot.empty:
            foreign_days, foreign_dir = consecutive_count(pivot["外資"])
            latest_inst_total = int(pivot["合計"].iloc[0])
            latest_foreign = int(pivot["外資"].iloc[0])
            latest_trust = int(pivot["投信"].iloc[0])
            latest_dealer = int(pivot["自營商"].iloc[0])
        else:
            foreign_days, foreign_dir = 0, "中性"
            latest_inst_total = latest_foreign = latest_trust = latest_dealer = 0

        # --- 5. 抓月營收 (ETF 跳過) ---
        latest_yoy = latest_mom = latest_revenue = 0
        has_revenue = False

        if not is_etf:
            try:
                rev_start = (df["date"].max() - pd.Timedelta(days=550)).strftime("%Y-%m-%d")
                revenue = dl.taiwan_stock_month_revenue(
                    stock_id=stock_id, start_date=rev_start, end_date=end_date
                )
                if not revenue.empty:
                    revenue["date"] = pd.to_datetime(revenue["date"])
                    revenue = revenue.sort_values("date").reset_index(drop=True)
                    revenue["MoM_%"] = revenue["revenue"].pct_change(periods=1) * 100
                    revenue["YoY_%"] = revenue["revenue"].pct_change(periods=12) * 100

                    latest_rev = revenue.iloc[-1]
                    latest_yoy = float(latest_rev["YoY_%"]) if pd.notna(latest_rev["YoY_%"]) else 0
                    latest_mom = float(latest_rev["MoM_%"]) if pd.notna(latest_rev["MoM_%"]) else 0
                    latest_revenue = float(latest_rev["revenue"]) / 100000000
                    has_revenue = True
            except Exception:
                pass

        # --- 6. 警示彙整 ---
        alerts = {"red": [], "yellow": [], "green": []}

        if pd.notna(latest["RSI14"]):
            if latest["RSI14"] > 80:
                alerts["red"].append("RSI 嚴重超買")
            elif latest["RSI14"] > 70:
                alerts["yellow"].append("RSI 接近超買")
            elif latest["RSI14"] < 30:
                alerts["green"].append("RSI 超賣可能反彈")

        if pd.notna(latest["K"]) and pd.notna(latest["D"]):
            if latest["K"] > 80 and latest["D"] > 80:
                alerts["red"].append("KD 高檔鈍化")
            elif latest["K"] < 20 and latest["D"] < 20:
                alerts["green"].append("KD 低檔鈍化")

            if len(df) >= 2:
                prev_k, prev_d = df["K"].iloc[-2], df["D"].iloc[-2]
                if pd.notna(prev_k) and pd.notna(prev_d):
                    if latest["K"] < latest["D"] and prev_k > prev_d:
                        alerts["red"].append("KD 死叉")
                    elif latest["K"] > latest["D"] and prev_k < prev_d:
                        alerts["green"].append("KD 黃金交叉")

        if pd.notna(latest["MA5"]) and pd.notna(latest["MA20"]) and pd.notna(latest["MA60"]):
            if latest["close"] > latest["MA5"] > latest["MA20"] > latest["MA60"]:
                alerts["green"].append("均線多頭排列")
            elif latest["close"] < latest["MA5"] < latest["MA20"] < latest["MA60"]:
                alerts["red"].append("均線空頭排列")

        if pd.notna(latest["Volume_Ratio"]):
            if latest["Volume_Ratio"] > 2:
                alerts["yellow"].append(f"爆量 ({latest['Volume_Ratio']:.1f}x)")
            elif latest["Volume_Ratio"] < 0.5:
                alerts["yellow"].append("量縮警示")

        latest_change = float(latest["Change_Pct"]) if pd.notna(latest["Change_Pct"]) else 0
        if latest_change > 0 and latest_inst_total < 0:
            alerts["red"].append("籌碼背離")
        elif latest_change < 0 and latest_inst_total > 0:
            alerts["green"].append("法人逆勢買進")

        if foreign_days >= 3:
            if foreign_dir == "買超":
                alerts["green"].append(f"外資連{foreign_days}買")
            else:
                alerts["red"].append(f"外資連{foreign_days}賣")

        if has_revenue:
            if latest_yoy > 30:
                alerts["green"].append(f"營收YoY +{latest_yoy:.0f}%")
            elif latest_yoy > 10:
                alerts["green"].append(f"營收YoY +{latest_yoy:.0f}%")
            elif latest_yoy < -10:
                alerts["red"].append(f"營收YoY {latest_yoy:.0f}%")

        # --- 7. 評分 ---
        tech_score = 3
        if pd.notna(latest["RSI14"]):
            if latest["RSI14"] > 80 or (pd.notna(latest["K"]) and pd.notna(latest["D"]) and latest["K"] > 80 and latest["D"] > 80):
                tech_score = 5
            elif latest["RSI14"] > 70:
                tech_score = 4
            elif pd.notna(latest["MA5"]) and pd.notna(latest["MA20"]) and latest["close"] > latest["MA5"] > latest["MA20"]:
                tech_score = 4
            elif pd.notna(latest["MA20"]) and latest["close"] < latest["MA20"]:
                tech_score = 2

        chip_score = 3
        if latest_change > 0 and latest_inst_total < 0:
            chip_score = 2
        elif foreign_days >= 3 and foreign_dir == "買超":
            chip_score = 5
        elif foreign_days >= 3 and foreign_dir == "賣超":
            chip_score = 2
        elif latest_inst_total > 0:
            chip_score = 4

        if has_revenue:
            if latest_yoy > 30:
                fund_score = 5
            elif latest_yoy > 10:
                fund_score = 4
            elif latest_yoy > 0:
                fund_score = 3
            elif latest_yoy > -10:
                fund_score = 2
            else:
                fund_score = 1
        else:
            fund_score = 3  # ETF 預設中性

        # --- 8. 整體狀態 ---
        if len(alerts["red"]) >= 2:
            overall_status = "🔴 過熱"
        elif len(alerts["red"]) >= 1:
            overall_status = "🟡 觀察"
        elif len(alerts["green"]) >= 2:
            overall_status = "🟢 健康"
        else:
            overall_status = "⚪ 中性"

        # --- 9. 回傳完整報告 ---
        return {
            "股票代號": stock_id,
            "股票名稱": stock_name,
            "is_etf": is_etf,
            "has_revenue": has_revenue,
            "df": df,
            "pivot": pivot,
            "收盤價": float(latest["close"]),
            "漲跌幅": latest_change,
            "成交量(張)": int(latest["volume"] / 1000),
            "RSI": round(float(latest["RSI14"]), 2) if pd.notna(latest["RSI14"]) else None,
            "K": round(float(latest["K"]), 2) if pd.notna(latest["K"]) else None,
            "D": round(float(latest["D"]), 2) if pd.notna(latest["D"]) else None,
            "MACD": round(float(latest["MACD"]), 2) if pd.notna(latest["MACD"]) else None,
            "MA5": round(float(latest["MA5"]), 2) if pd.notna(latest["MA5"]) else None,
            "MA20": round(float(latest["MA20"]), 2) if pd.notna(latest["MA20"]) else None,
            "MA60": round(float(latest["MA60"]), 2) if pd.notna(latest["MA60"]) else None,
            "量比": round(float(latest["Volume_Ratio"]), 2) if pd.notna(latest["Volume_Ratio"]) else None,
            "外資(張)": latest_foreign,
            "投信(張)": latest_trust,
            "自營商(張)": latest_dealer,
            "法人合計(張)": latest_inst_total,
            "外資連續": f"{foreign_days}日{foreign_dir}",
            "最新月營收(億)": round(latest_revenue, 2),
            "營收YoY(%)": round(latest_yoy, 2),
            "營收MoM(%)": round(latest_mom, 2),
            "技術面評分": tech_score,
            "籌碼面評分": chip_score,
            "基本面評分": fund_score,
            "整體狀態": overall_status,
            "警示_紅": alerts["red"],
            "警示_黃": alerts["yellow"],
            "警示_綠": alerts["green"],
        }

    except Exception as e:
        return {"股票代號": stock_id, "錯誤": str(e)}


# ============================================
# 繪製 K 線圖
# ============================================
def plot_kline(df, stock_name, stock_id):
    """繪製互動式 K 線圖（含均線、成交量、RSI、MACD、KD）"""
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.45, 0.12, 0.14, 0.14, 0.15],
        subplot_titles=("日 K 線 + 均線", "成交量", "RSI(14)", "MACD(12,26,9)", "KD(9,3,3)")
    )

    # K 線
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color="red", decreasing_line_color="green",
        name="K線"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["date"], y=df["MA5"], name="MA5",
                             line=dict(color="yellow", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA20"], name="MA20",
                             line=dict(color="orange", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["MA60"], name="MA60",
                             line=dict(color="magenta", width=1)), row=1, col=1)

    # 成交量
    colors = ["red" if c >= o else "green" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], marker_color=colors,
                         name="成交量", showlegend=False), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df["date"], y=df["RSI14"], name="RSI",
                             line=dict(color="yellow", width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # MACD
    if "MACD_hist" in df.columns and df["MACD_hist"].notna().any():
        hist_colors = ["red" if v >= 0 else "green" for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df["date"], y=df["MACD_hist"], marker_color=hist_colors,
                             name="MACD柱", showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MACD"], name="DIF",
                                 line=dict(color="cyan", width=1.5)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MACD_signal"], name="DEA",
                                 line=dict(color="yellow", width=1.5)), row=4, col=1)

    # KD
    fig.add_trace(go.Scatter(x=df["date"], y=df["K"], name="K",
                             line=dict(color="cyan", width=1.5)), row=5, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["D"], name="D",
                             line=dict(color="magenta", width=1.5)), row=5, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=5, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=5, col=1)

    fig.update_layout(
        title=f"{stock_name} ({stock_id}) - 技術分析",
        template="plotly_dark",
        height=800,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        hovermode="x unified"
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


def plot_institutional(pivot, df):
    """繪製法人籌碼柱狀圖（含股價疊圖）"""
    if pivot.empty:
        return None

    recent = pivot.head(10).sort_index()
    price_recent = df[df["date"].isin(pd.to_datetime(recent.index))][["date", "close"]]
    price_recent = price_recent.sort_values("date")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(x=recent.index, y=recent["外資"],
                         name="外資", marker_color="#3b82f6"), secondary_y=False)
    fig.add_trace(go.Bar(x=recent.index, y=recent["投信"],
                         name="投信", marker_color="#ef4444"), secondary_y=False)
    fig.add_trace(go.Bar(x=recent.index, y=recent["自營商"],
                         name="自營商", marker_color="#a855f7"), secondary_y=False)
    fig.add_trace(go.Bar(x=recent.index, y=recent["合計"],
                         name="合計", marker_color="#22c55e"), secondary_y=False)
    fig.add_trace(go.Scatter(x=price_recent["date"], y=price_recent["close"],
                             name="股價", mode="lines+markers",
                             line=dict(color="yellow", width=2),
                             marker=dict(size=8)), secondary_y=True)

    fig.update_layout(
        title="三大法人近10日籌碼動向",
        template="plotly_dark",
        barmode="group",
        height=400,
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="法人買賣超(張)", secondary_y=False)
    fig.update_yaxes(title_text="股價(元)", secondary_y=True)
    return fig


# ============================================
# 卡片渲染
# ============================================
def render_card(report, note=""):
    """渲染單一股票卡片"""
    if "錯誤" in report:
        return f"""
        <div class="status-card">
            <h4 style="color: #ff6b6b;">❌ {report['股票代號']}</h4>
            <p style="color: #aaa;">{report['錯誤']}</p>
        </div>
        """

    status = report["整體狀態"]
    if "🔴" in status:
        status_color = "#ff6b6b"
        status_bg = "#4a1a1a"
    elif "🟡" in status:
        status_color = "#ffd93d"
        status_bg = "#4a3a1a"
    elif "🟢" in status:
        status_color = "#6bcf7f"
        status_bg = "#1a4a2a"
    else:
        status_color = "#aaa"
        status_bg = "#2a2a2a"

    chg = report["漲跌幅"]
    if chg > 0:
        chg_html = f'<span style="color:#ff6b6b;">▲ {chg:+.2f}%</span>'
    elif chg < 0:
        chg_html = f'<span style="color:#6bcf7f;">▼ {chg:+.2f}%</span>'
    else:
        chg_html = '<span style="color:#aaa;">─ 0.00%</span>'

    alerts_html = ""
    for a in report["警示_紅"]:
        alerts_html += f'<span class="alert-red">🔴 {a}</span>'
    for a in report["警示_黃"]:
        alerts_html += f'<span class="alert-yellow">🟡 {a}</span>'
    for a in report["警示_綠"]:
        alerts_html += f'<span class="alert-green">🟢 {a}</span>'
    if not alerts_html:
        alerts_html = '<span style="color:#aaa;">⚪ 無特殊訊號</span>'

    rsi = report.get("RSI") or "N/A"
    k = report.get("K") or "N/A"
    d = report.get("D") or "N/A"
    foreign = report.get("外資(張)", 0)
    foreign_color = "#ff6b6b" if foreign > 0 else "#6bcf7f" if foreign < 0 else "#aaa"

    yoy = report.get("營收YoY(%)", 0)
    if report.get("has_revenue"):
        yoy_color = "#ff6b6b" if yoy > 0 else "#6bcf7f"
        yoy_str = f'<span style="color:{yoy_color}">{yoy:+.1f}%</span>'
    else:
        yoy_str = '<span style="color:#aaa">ETF 無營收</span>'

    note_html = f'<p style="color:#888; font-size:11px; margin-top:6px;">📝 {note}</p>' if note else ""

    return f"""
    <div class="status-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h4 style="margin:0; color:#fff;">{report['股票名稱']} <span style="color:#888;">{report['股票代號']}</span></h4>
                <p style="margin:4px 0; font-size:18px; font-weight:bold;">
                    {report['收盤價']} {chg_html}
                </p>
            </div>
            <div style="background:{status_bg}; color:{status_color};
                        padding:6px 12px; border-radius:6px; font-weight:bold;">
                {status}
            </div>
        </div>

        <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:6px; margin:12px 0; font-size:12px;">
            <div style="background:#2a2a2a; padding:6px 10px; border-radius:4px;">
                <span style="color:#aaa;">RSI</span>
                <span style="float:right; font-weight:500;">{rsi}</span>
            </div>
            <div style="background:#2a2a2a; padding:6px 10px; border-radius:4px;">
                <span style="color:#aaa;">KD</span>
                <span style="float:right; font-weight:500;">{k}/{d}</span>
            </div>
            <div style="background:#2a2a2a; padding:6px 10px; border-radius:4px;">
                <span style="color:#aaa;">外資(張)</span>
                <span style="float:right; color:{foreign_color}; font-weight:500;">{foreign:+,}</span>
            </div>
            <div style="background:#2a2a2a; padding:6px 10px; border-radius:4px;">
                <span style="color:#aaa;">營收YoY</span>
                <span style="float:right; font-weight:500;">{yoy_str}</span>
            </div>
        </div>

        <div style="font-size:11px; color:#aaa; margin-bottom:8px;">
            技術 {report['技術面評分']}/5 ⭐ |
            籌碼 {report['籌碼面評分']}/5 ⭐ |
            基本面 {report['基本面評分']}/5 ⭐
        </div>

        <div>
            {alerts_html}
        </div>
        {note_html}
    </div>
    """


# ============================================
# 主介面
# ============================================
def main():
    st.title("📊 台股觀察清單分析")
    st.caption(f"⏰ 最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 初始化 session state
    if "watchlist_data" not in st.session_state:
        st.session_state.watchlist_data = load_watchlist()
    if "all_reports" not in st.session_state:
        st.session_state.all_reports = []

    # === 側邊欄 ===
    with st.sidebar:
        st.header("⚙️ 操作面板")

        if st.button("🔄 全部分析", type="primary", use_container_width=True):
            stocks = st.session_state.watchlist_data["stocks"]
            progress = st.progress(0, "準備分析...")
            reports = []
            for i, s in enumerate(stocks):
                progress.progress((i + 1) / len(stocks),
                                  f"分析 {s['name']} ({s['id']})...")
                r = analyze_stock(s["id"])
                r["備註"] = s.get("note", "")
                reports.append(r)
                time.sleep(0.3)
            st.session_state.all_reports = reports
            progress.empty()
            st.success(f"✅ 完成 {len(reports)} 檔分析")

        st.divider()

        st.subheader("➕ 新增股票")
        new_id = st.text_input("代號", placeholder="例: 2330")
        new_name = st.text_input("名稱", placeholder="例: 台積電")
        new_note = st.text_input("備註", placeholder="例: 庫存")
        if st.button("加入清單", use_container_width=True):
            if new_id and new_name:
                st.session_state.watchlist_data["stocks"].append({
                    "id": new_id, "name": new_name, "note": new_note
                })
                st.success(f"已加入 {new_name}")
                st.warning("⚠️ 重新整理或按全部分析才會更新")
            else:
                st.error("請填寫代號和名稱")

        st.divider()

        st.subheader("🗑️ 移除股票")
        stocks = st.session_state.watchlist_data["stocks"]
        if stocks:
            options = [f"{s['id']} - {s['name']}" for s in stocks]
            to_remove = st.selectbox("選擇要移除的", options)
            if st.button("確認移除", use_container_width=True):
                idx = options.index(to_remove)
                removed = st.session_state.watchlist_data["stocks"].pop(idx)
                st.success(f"已移除 {removed['name']}")

        st.divider()

        if st.session_state.all_reports:
            st.subheader("📥 匯出 Excel")
            export_data = []
            for r in st.session_state.all_reports:
                if "錯誤" not in r:
                    export_data.append({
                        "代號": r["股票代號"],
                        "名稱": r["股票名稱"],
                        "備註": r.get("備註", ""),
                        "狀態": r["整體狀態"],
                        "收盤": r["收盤價"],
                        "漲跌%": r["漲跌幅"],
                        "RSI": r["RSI"],
                        "K": r["K"],
                        "D": r["D"],
                        "外資(張)": r["外資(張)"],
                        "投信(張)": r["投信(張)"],
                        "營收YoY%": r["營收YoY(%)"],
                        "技術評分": r["技術面評分"],
                        "籌碼評分": r["籌碼面評分"],
                        "基本面評分": r["基本面評分"],
                        "紅色警示": " | ".join(r["警示_紅"]),
                        "綠色正面": " | ".join(r["警示_綠"]),
                    })
            if export_data:
                df_export = pd.DataFrame(export_data)
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df_export.to_excel(writer, index=False, sheet_name="觀察清單")
                st.download_button(
                    label="📥 下載 Excel",
                    data=buffer.getvalue(),
                    file_name=f"觀察清單_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

    # === 主畫面 ===
    if not st.session_state.all_reports:
        st.info("👈 請點擊左側「🔄 全部分析」開始")
        st.subheader("📋 目前觀察清單")
        for s in st.session_state.watchlist_data["stocks"]:
            note = f" - {s['note']}" if s.get("note") else ""
            st.write(f"• **{s['name']}** ({s['id']}){note}")
        return

    # 統計
    valid_reports = [r for r in st.session_state.all_reports if "錯誤" not in r]
    red_count = sum(1 for r in valid_reports if "🔴" in r["整體狀態"])
    yellow_count = sum(1 for r in valid_reports if "🟡" in r["整體狀態"])
    green_count = sum(1 for r in valid_reports if "🟢" in r["整體狀態"])
    gray_count = sum(1 for r in valid_reports if "⚪" in r["整體狀態"])

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📊 總數", len(valid_reports))
    col2.metric("🔴 過熱", red_count)
    col3.metric("🟡 觀察", yellow_count)
    col4.metric("🟢 健康", green_count)
    col5.metric("⚪ 中性", gray_count)

    # 警示彙總
    if red_count > 0 or yellow_count > 0:
        with st.expander("⚠️ 今日警示彙整", expanded=True):
            for r in valid_reports:
                if r["警示_紅"]:
                    st.error(f"**{r['股票名稱']} ({r['股票代號']})**："
                             f"{' | '.join(r['警示_紅'])}")
                elif r["警示_黃"]:
                    st.warning(f"**{r['股票名稱']} ({r['股票代號']})**："
                               f"{' | '.join(r['警示_黃'])}")

    # 分頁
    tab1, tab2, tab3 = st.tabs(["🎴 卡片總覽", "📊 詳細表格", "📈 單檔分析"])

    with tab1:
        cols = st.columns(2)
        for i, r in enumerate(st.session_state.all_reports):
            with cols[i % 2]:
                st.markdown(render_card(r, r.get("備註", "")), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

    with tab2:
        table_data = []
        for r in valid_reports:
            table_data.append({
                "代號": r["股票代號"],
                "名稱": r["股票名稱"],
                "狀態": r["整體狀態"],
                "收盤": r["收盤價"],
                "漲跌%": f"{r['漲跌幅']:+.2f}",
                "RSI": r["RSI"],
                "K/D": f"{r['K']}/{r['D']}" if r["K"] and r["D"] else "N/A",
                "外資(張)": f"{r['外資(張)']:+,}",
                "營收YoY%": f"{r['營收YoY(%)']:+.1f}%" if r["has_revenue"] else "N/A",
                "技/籌/基": f"{r['技術面評分']}/{r['籌碼面評分']}/{r['基本面評分']}",
                "警示": " | ".join(r["警示_紅"][:2]) if r["警示_紅"] else "—",
            })
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

    with tab3:
        if valid_reports:
            options = [f"{r['股票名稱']} ({r['股票代號']})" for r in valid_reports]
            selected = st.selectbox("選擇股票", options)
            idx = options.index(selected)
            r = valid_reports[idx]

            st.subheader(f"{r['股票名稱']} ({r['股票代號']})")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("收盤價", r["收盤價"], f"{r['漲跌幅']:+.2f}%")
            c2.metric("RSI", r["RSI"] or "N/A")
            c3.metric("外資(張)", f"{r['外資(張)']:+,}")
            c4.metric("整體狀態", r["整體狀態"])

            if "df" in r and not r["df"].empty:
                st.plotly_chart(plot_kline(r["df"], r["股票名稱"], r["股票代號"]),
                                use_container_width=True)

            if "pivot" in r and not r["pivot"].empty:
                inst_fig = plot_institutional(r["pivot"], r["df"])
                if inst_fig:
                    st.plotly_chart(inst_fig, use_container_width=True)

                st.subheader("💼 三大法人近10日明細 (張)")
                st.dataframe(r["pivot"].head(10), use_container_width=True)


if __name__ == "__main__":
    main()
