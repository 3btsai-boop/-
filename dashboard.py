import streamlit as st
import pandas as pd
import plotly.express as px
import jieba
import jieba.analyse
from datetime import datetime, timedelta
import os
import time
import base64

# --- 0. 全域設定 ---
st.set_page_config(
    page_title="義享天地輿情戰情室 V15",
    page_icon="🏙️",
    layout="wide"
)

# --- CSS 美化 (Flexbox 剛性佈局 - V9.0 架構保持不變) ---
st.markdown("""
    <style>
    .block-container {
        padding-top: 3.5rem; 
        padding-bottom: 2rem;
    }
    
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .stPlotlyChart {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        padding: 10px;
    }
    
    .header-container {
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: flex-start;
        gap: 25px;
        margin-bottom: 20px;
        width: 100%;
    }
    
    .logo-img {
        height: 85px;
        width: auto;
        object-fit: contain;
        flex-shrink: 0;
    }
    
    .title-box {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
        color: #000;
    }
    
    .sub-title {
        font-size: 1rem;
        color: #666;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. 核心邏輯：情緒計分引擎 (V15.0 競品黑名單強化版) ---
class SentimentEngine:
    def __init__(self):
        # 1. [絕對語意] 出現即定調 (優先級最高)
        self.deadly_negative_patterns = [
            "不會再來", "一次店", "再也不", "勸退", "不想再", "絕不", 
            "爛死", "爛透", "氣死", "拒絕", "黑名單", "浪費錢", 
            "浪費時間", "最爛", "真的很糟", "無法接受", "沒下次", 
            "不予置評", "不推", "不優", "不如去", "還不如", "寧願去", 
            "輸給", "慘輸", "被屌打", "笑死", "笑爛", "傻眼", "無言", "誇張", "悲劇"
        ]
        
        self.super_positive_patterns = [
            "必回訪", "一定會再", "一定再", "唯一推薦", "神店", 
            "最愛", "超愛", "很頂", "沒對手", "第一名", "滿分",
            "一定會再去", "舒服", "很好逛", "好逛", "超好逛"
        ]

        # 2. [關鍵字權重] (競品扣分加重至 -3)
        self.neg_words = {
            # 設施抱怨
            'B4': -2, 'B5': -4, 'B6': -4, 'B7': -5, 
            '停車': -3, '出口': -3, '動線': -4, '塞車': -4, '塞爆': -5,
            '排隊': -3, '等很久': -3, '卡住': -3, '迷宮': -4,
            # 情緒詞
            '爛': -5, '差': -4, '失望': -4, '難吃': -4, '髒': -4, '噁心': -5,
            '盤子': -5, '智障': -5, '廢': -4, '抵制': -5, '火大': -4, 
            '雷': -5, '糟糕': -4, '後悔': -4, '不行': -3, '普通': -2,
            # 競品黑名單 (只要提到對手，通常都是在貶低義享，扣分加重)
            '巨蛋': -3, '漢神': -3, '夢時代': -3, '好市多': -3, 'Costco': -3, 
            '遠百': -3, '新光': -3, '三越': -3, '草衙道': -2, '高鐵': -1
        }
        
        self.pos_words = {
            # 正面詞彙 (權重加重，保護好評)
            '好吃': 5, '寬敞': 4, '喜歡': 4, '推薦': 5, '必吃': 5,
            '漂亮': 3, '質感': 3, '開心': 3, '棒': 4, '優': 4,
            '讚': 5, '推': 3, '不錯': 3, '愛': 4, '勝': 3, '贏': 3,
            '優惠': 2, '折抵': 2, '方便': 3, '大': 2, '新': 2, 
            '旭集': 4, '饗泰多': 4, '問鼎': 3, '京翠': 3
        }
        
        self.negation_words = ['不', '沒', '無', '非', '別', '不會', '不用', '不太']

    def analyze(self, text):
        if not isinstance(text, str): return "中性"
        text = text.strip()
        
        # 1. 絕對快篩
        for pattern in self.deadly_negative_patterns:
            if pattern in text: return "負面"
        for pattern in self.super_positive_patterns:
            if pattern in text: return "正面"

        # 2. 前處理
        base_score = 0
        if "[推]" in text: base_score += 1
        if "[噓]" in text: base_score -= 4
        
        clean_text = text.replace("[推]", "").replace("[噓]", "").replace("[→]", "").replace("[標題]", "")
        
        # 3. 關鍵字計分
        score = base_score
        words = jieba.lcut(clean_text)
        
        for i, word in enumerate(words):
            word_score = 0
            if word in self.neg_words:
                word_score = self.neg_words[word]
            elif word in self.pos_words:
                word_score = self.pos_words[word]
            if i > 0 and words[i-1] in self.negation_words:
                word_score = -word_score
            score += word_score
        
        # 4. 判定門檻
        if score <= -1: return "負面"
        elif score >= 2: return "正面"
        else: return "中性"

sentiment_engine = SentimentEngine()

# --- 2. 數據處理 ---
def solve_future_date_issue(df):
    now = datetime.now()
    cutoff = now + timedelta(days=1)
    def adjust_date(x):
        try:
            d = pd.to_datetime(x) if isinstance(x, str) else x
            if pd.isnull(d): return d
            if d > cutoff: return d.replace(year=d.year - 1)
            return d
        except: return x
    df['date'] = df['date'].apply(adjust_date)
    return df

@st.cache_data(ttl=60)
def load_data(csv_path="my_data.csv"):
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['date', 'source', 'content', 'link', 'sentiment'])
    df = pd.read_csv(csv_path)
    
    # 🚨【關鍵操作】強制捨棄 CSV 裡可能的舊標籤，使用 V15 引擎重算
    if 'sentiment' in df.columns:
        del df['sentiment']
        
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = solve_future_date_issue(df)
    
    # 使用 V15 引擎重新計算
    df['sentiment'] = df['content'].apply(sentiment_engine.analyze)
    return df

# --- 3. 爬蟲整合 (已修復：解決 NoneType 錯誤) ---
def run_spider_pipeline():
    # 定義機器人變數，避免未初始化錯誤
    bot = None
    with st.spinner('🚀 正在啟動爬蟲衛星，全速更新中...'):
        try:
            # 1. 匯入您的爬蟲檔案 (必須是 history_spider_final.py)
            import history_spider_final as spider_module
            
            # 2. 初始化爬蟲類別
            bot = spider_module.EskyHistorySpiderV10()
            
            # 🚨【關鍵修正】強制啟動瀏覽器驅動程式 (Driver)
            # 這行代碼解決了 'NoneType' object has no attribute 'get' 的問題
            bot.driver = bot._init_selenium()
            
            # 3. 開始爬取 (瀏覽器視窗會跳出來，請勿關閉)
            # 如果 Mobile01 遇到 Cloudflare 驗證，請手動在跳出的視窗點擊
            bot.crawl_ptt()
            bot.crawl_mobile01()
            bot.crawl_dcard()
            
            # 4. 取得新資料
            new_data = bot.data_list
            
            # 5. 爬取完成，關閉瀏覽器釋放記憶體
            bot.close()
            
            # 6. 資料合併與存檔
            if new_data:
                if os.path.exists("my_data.csv"):
                    old_df = pd.read_csv("my_data.csv")
                    # 確保舊資料有被讀取，並與新資料合併
                    final_df = pd.concat([old_df, pd.DataFrame(new_data)])
                else:
                    final_df = pd.DataFrame(new_data)
                    
                # 以內容去重 (避免重複推文)
                final_df.drop_duplicates(subset=['content'], keep='last', inplace=True)
                final_df.to_csv("my_data.csv", index=False, encoding='utf-8-sig')
                st.success(f"✅ 更新成功！共收集 {len(new_data)} 筆新資料。")
                time.sleep(2)
                st.rerun()
            else:
                st.warning("⚠️ 爬蟲執行完成，但未發現新資料。")
                
        except Exception as e:
            # 發生錯誤時確保瀏覽器關閉
            if bot and bot.driver:
                bot.close()
            st.error(f"更新失敗: {str(e)}")

# --- 4. 輔助函數 ---
def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- 5. 圖表繪製 ---

def plot_clean_trend(df, freq_opt, start_dt, end_dt):
    freq_map = {'日 (Day)': 'D', '週 (Week)': 'W', '月 (Month)': 'M'}
    freq_code = freq_map[freq_opt]
    
    all_dates = pd.date_range(start=start_dt, end=end_dt, freq=freq_code)
    sentiments = ['正面', '負面', '中性']
    full_idx = pd.MultiIndex.from_product([all_dates, sentiments], names=['date', 'sentiment'])
    full_df = pd.DataFrame(index=full_idx).reset_index()
    
    raw_trend = df.groupby([pd.Grouper(key='date', freq=freq_code), 'sentiment']).size().reset_index(name='count')
    trend = pd.merge(full_df, raw_trend, on=['date', 'sentiment'], how='left')
    trend['count'] = trend['count'].fillna(0)
    
    colors = {'正面': '#00b894', '負面': '#d63031', '中性': '#b2bec3'}
    
    fig = px.line(
        trend, x='date', y='count', color='sentiment',
        color_discrete_map=colors,
        render_mode='svg'
    )
    
    total_points = len(trend)
    mode_setting = "lines" if total_points > 120 else "lines+markers"

    fig.update_traces(
        mode=mode_setting, 
        line_shape="spline", 
        line_width=2.5,
        marker_size=7,
        hovertemplate='%{y} 篇'
    )
    
    delta_days = (end_dt - start_dt).days
    tick_fmt = "%Y-%m" if delta_days > 365 else "%Y-%m-%d"

    fig.update_layout(
        title="",
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode="x unified",
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="left", 
            x=0, 
            title=""
        ),
        xaxis=dict(
            title="",
            showgrid=False,
            range=[start_dt, end_dt],
            tickformat=tick_fmt,
            nticks=12,
            tickangle=0,
            linecolor='#dfe6e9'
        ),
        yaxis=dict(
            title="聲量 (篇)",
            showgrid=True,
            gridcolor='#f1f2f6',
            zeroline=False
        ),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig

def plot_clean_bar(df_kw, color):
    fig = px.bar(
        df_kw, x='權重', y='關鍵詞', orientation='h',
        text='權重'
    )
    fig.update_traces(
        marker_color=color,
        texttemplate='%{text:.1f}', 
        textposition='outside',
        width=0.65
    )
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(visible=False),
        yaxis=dict(categoryorder='total ascending', title=""),
        margin=dict(l=0, r=40, t=30, b=0),
        height=320,
        font=dict(size=14)
    )
    return fig

# --- 6. 主程式 ---

with st.sidebar:
    st.header("⚙️ 監測控制台")
    if st.button("🚀 啟動即時更新", type="primary"):
        run_spider_pipeline()
    st.markdown("---")
    
    df = load_data()
    if df.empty:
        st.warning("⚠️ 暫無數據")
        st.stop()
        
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    target_start = datetime(2021, 1, 1).date()
    default_start = target_start if min_date <= target_start else min_date
    
    st.caption("📅 日期篩選")
    date_range = st.date_input("", [default_start, max_date])

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_dt, end_dt = date_range
    mask = (df['date'].dt.date >= start_dt) & (df['date'].dt.date <= end_dt)
    df_filtered = df.loc[mask]
else:
    st.info("請選擇完整的日期起訖。")
    st.stop()

if df_filtered.empty:
    st.warning("此區間無數據")
    st.stop()

# --- Header Area (V9.0 Flexbox 架構) ---

img_tag = ""
if os.path.exists("logo.png"):
    img_b64 = get_img_as_base64("logo.png")
    img_tag = f'<img src="data:image/png;base64,{img_b64}" class="logo-img">'
else:
    img_tag = '<img src="https://www.esky-land.com.tw/img/logo.png" class="logo-img">'

st.markdown(f"""
    <div class="header-container">
        {img_tag}
        <div class="title-box">
            <h1 class="main-title">義享天地・輿情戰情室</h1>
            <div class="sub-title">Data Range: <b>{start_dt}</b> ~ <b>{end_dt}</b></div>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# KPI
neg_df = df_filtered[df_filtered['sentiment'] == '負面']
pos_df = df_filtered[df_filtered['sentiment'] == '正面']
k1, k2, k3, k4 = st.columns(4)
k1.metric("📦 總聲量", f"{len(df_filtered)}")
k2.metric("😡 負評數", f"{len(neg_df)}", delta_color="inverse")
k3.metric("🥰 好評數", f"{len(pos_df)}")
k4.metric("📊 負評率", f"{(len(neg_df)/len(df_filtered)*100):.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
t1, t2 = st.tabs(["📊 趨勢分析", "⚔️ 關鍵字對決"])

with t1:
    day_diff = (end_dt - start_dt).days
    idx = 0
    if day_diff > 365: idx = 2 
    elif day_diff > 60: idx = 1 
    
    col_opt, _ = st.columns([2, 5])
    with col_opt:
        freq_opt = st.radio("檢視粒度:", ['日 (Day)', '週 (Week)', '月 (Month)'], index=idx, horizontal=True)

    fig_trend = plot_clean_trend(df_filtered, freq_opt, start_dt, end_dt)
    st.plotly_chart(fig_trend, use_container_width=True)
    
    with st.expander("查看來源分佈"):
        fig_pie = px.pie(df_filtered, names='source', hole=0.6, color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_pie, use_container_width=True)

with t2:
    c_neg, c_pos = st.columns(2)
    
    # 🚨【關鍵修正】停用詞大清洗：濾除「這種」、「那個」、「比較」等無意義詞
    stop_words = set([
        "高雄", "義享", "天地", "百貨", "巨蛋", "感覺", "比較", "真的", "現在", "今天", "時候", "知道", "看到", 
        "有的", "沒有", "什麼", "可以", "一個", "就是", "還是", "我們", "你們", "因為", "可能", "其實", "覺得", 
        "不過", "這個", "那個", "去過", "大家", "請問", "問題", "閒聊", "新聞", "分享", "文章", "作者", "標題", 
        "時間", "原本", "以為", "結果", "部分", "目前", "已經", "怎麼", "這樣", "最近", "這家", "這種", "那種",
        "一樣", "一點", "一下", "一直", "只是", "但是", "然後", "還有", "只是", "甚至", "而且", "不如", "如果"
    ])
    if os.path.exists("stop_words.txt"):
        with open("stop_words.txt", "r", encoding="utf-8") as f:
            for line in f: stop_words.add(line.strip())

    def get_kw_df(texts):
        full = " ".join([str(t) for t in texts])
        tags = jieba.analyse.extract_tags(full, topK=80, withWeight=True)
        filtered = [(w, s) for w, s in tags if w not in stop_words and len(w)>1 and not w.isdigit()]
        return pd.DataFrame(filtered[:8], columns=['關鍵詞', '權重'])

    with c_neg:
        st.markdown("#### 😡 負面痛點")
        if not neg_df.empty:
            kw_neg = get_kw_df(neg_df['content'].tolist())
            if not kw_neg.empty:
                st.plotly_chart(plot_clean_bar(kw_neg, '#d63031'), use_container_width=True)
            with st.expander("查看負評列表"):
                st.dataframe(neg_df[['date','source','content']], hide_index=True)
        else: st.info("無數據")

    with c_pos:
        st.markdown("#### 🥰 正面亮點")
        if not pos_df.empty:
            kw_pos = get_kw_df(pos_df['content'].tolist())
            if not kw_pos.empty:
                st.plotly_chart(plot_clean_bar(kw_pos, '#00b894'), use_container_width=True)
            with st.expander("查看好評列表"):
                st.dataframe(pos_df[['date','source','content']], hide_index=True)
        else: st.info("無數據")

st.markdown("---")
st.caption(f"System v15.0 (Rule-Based Restoration) | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")