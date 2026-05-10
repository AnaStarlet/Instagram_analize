import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from analyzer import (
    analyze_sentiment,
    calculate_priority_score,
    classify_theme,
    generate_insights
)

st.set_page_config(
    page_title="Insight Search",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>

:root {

    --accent: #E75480;
    --accent2: #FF8FB1;
    --accent3: #FFB7CE;

    --soft-pink: #FFF1F5;
    --soft-pink-2: #FFE4EC;

    --dark-pink: #C94F7C;

    --border-light: rgba(231,84,128,0.12);

    --text: #2B2B2B;
}

html, body, [class*="css"] {

    font-family: 'Segoe UI';
}

body {

    background: linear-gradient(
        180deg,
        #FFF8FA,
        #FFF1F5
    );
}

[data-testid="collapsedControl"] {

    display: none;
}

section[data-testid="stSidebar"] {

    display: none;
}

.block-container {

    padding-top: 1.2rem;
}

h1 {

    color: var(--dark-pink);
}

.stTextInput input {

    border-radius: 24px !important;

    padding: 15px 22px !important;

    font-size: 17px !important;

    transition: 0.25s !important;

    border: 1px solid #FFD3E0 !important;

    background: white !important;
    
    color: black !important;
}

.stTextInput input:focus {

    border-color: var(--accent) !important;

    box-shadow: 0 0 0 0.2rem rgba(231,84,128,0.15) !important;
}

.stButton>button {

    background: linear-gradient(
        135deg,
        var(--accent),
        var(--accent2)
    ) !important;

    color: white !important;

    border-radius: 18px !important;

    border: none !important;

    height: 52px;

    font-weight: 600 !important;

    transition: 0.25s;
}

.stButton>button:hover {

    transform: translateY(-2px);

    box-shadow: 0 8px 24px rgba(231,84,128,0.22);
}

.stDownloadButton > button {

    background: linear-gradient(
        135deg,
        #F06292,
        #FF9EB8
    ) !important;

    color: white !important;

    border-radius: 18px !important;

    border: none !important;

    height: 48px;

    font-weight: 600 !important;
}

.stDownloadButton > button:hover {

    box-shadow: 0 8px 24px rgba(231,84,128,0.22);
}

div[role="radiogroup"] {

    justify-content: center;

    gap: 10px;
}

/* ЭТАЛОННЫЕ БЕЛЫЕ КВАДРАТЫ ДЛЯ СВЕТЛОЙ ТЕМЫ */
div[data-testid="stMetric"], .filter-box, .insight-box, .metric-card {
    background: linear-gradient(145deg, white, #FFF3F7) !important;
    border: 1px solid #FFD9E5 !important;
    border-radius: 18px !important;
    padding: 18px !important;
    box-shadow: 0 6px 20px rgba(231,84,128,0.06) !important;
}

.insight-box {
    border-left: 5px solid var(--accent) !important;
}

[data-testid="stDataFrame"] {

    border-radius: 20px;

    overflow: hidden;

    border: 1px solid #FFD6E2;

    box-shadow: 0 8px 24px rgba(231,84,128,0.06);
}

thead tr th {

    background-color: #FFE4EC !important;

    color: #8A2E52 !important;

    font-weight: 700 !important;
}

tbody tr:hover {

    background-color: #FFF3F7 !important;
}

.stSlider > div > div {

    color: var(--accent) !important;
}

.stSelectbox div[data-baseweb="select"] {

    border-radius: 16px !important;
}

.stCheckbox div[data-baseweb="checkbox"] > div {

    border-color: var(--accent) !important;
}

.stTabs [data-baseweb="tab"] {

    color: #C94F7C !important;

    font-weight: 600 !important;
}

.stTabs [aria-selected="true"] {

    background: linear-gradient(
        135deg,
        #FFE4EC,
        #FFF0F5
    ) !important;

    border-radius: 14px 14px 0 0 !important;
}

div[data-testid="stMetricLabel"] {
    color: #A63E68 !important;
}

div[data-testid="stMetricValue"] {
    color: #C94F7C !important;
}

/* --- ТОЛЬКО ДЛЯ ТЕМНОЙ ТЕМЫ: ЯРКО-РОЗОВЫЕ КВАДРАТЫ --- */
@media (prefers-color-scheme: dark) {
    div[data-testid="stMetric"], .filter-box, .insight-box, .metric-card {
        background: linear-gradient(145deg, #E75480, #C94F7C) !important; /* Яркий розовый, не бордовый */
        border: 1px solid #FF8FB1 !important;
    }
    div[data-testid="stMetricLabel"] > div > p {
        color: white !important;
    }
    div[data-testid="stMetricValue"] > div {
        color: white !important;
    }
}

</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():

    conn = sqlite3.connect("instagram_data.db")

    df = pd.read_sql(
        "SELECT * FROM comments",
        conn
    )

    conn.close()

    return df


comments_all = load_data()

st.markdown("""
<h1 style='
text-align:center;
font-size:56px;
font-weight:800;
margin-bottom:0;
'>
Insight Search
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<div style='
text-align:center;
opacity:0.7;
font-size:17px;
margin-bottom:28px;
'>
Instagram Content Analytics Platform
</div>
""", unsafe_allow_html=True)

if 'search_results' not in st.session_state:
    st.session_state['search_results'] = pd.DataFrame()

col1, col2, col3 = st.columns([1,4,1])

with col2:

    search_query = st.text_input(
        "Search",
        placeholder="Введите ключевые слова, URL или никнейм...",
        label_visibility="collapsed"
    )

    mode = st.radio(
        "Mode",
        [
            "🔑 По словам",
            "🔗 По URL",
            "👤 По автору",
            "✍️ Один текст"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    analyze_clicked = st.button(
        "Анализировать",
        use_container_width=True
    )

st.divider()

if analyze_clicked:

    if not search_query.strip():

        st.warning("Введите запрос")

    else:

        term = search_query.strip().lower()

        result = pd.DataFrame()

        if mode == "🔑 По словам":

            keys = [
                k.strip().lower()
                for k in search_query.split(',')
            ]

            result = comments_all[
                comments_all['comment_text']
                .astype(str)
                .str.lower()
                .apply(
                    lambda x:
                    any(k in x for k in keys)
                )
            ].copy()

        elif mode == "🔗 По URL":

            result = comments_all[
                comments_all['post_url']
                .astype(str)
                .str.contains(
                    term,
                    na=False,
                    case=False
                )
            ].copy()

        elif mode == "👤 По автору":

            result = comments_all[
                comments_all['comment_author']
                .astype(str)
                .str.lower()
                .str.contains(
                    term,
                    na=False
                )
            ].copy()

        elif mode == "✍️ Один текст":

            result = pd.DataFrame([{
                'comment_text': search_query,
                'comment_author': 'Вы',
                'comment_date': pd.Timestamp.now(),
                'post_url': ''
            }])

        if result.empty:

            st.error("Ничего не найдено")

        else:

            result['sentiment'] = analyze_sentiment(
                result['comment_text'].tolist()
            )

            result['theme'] = result[
                'comment_text'
            ].apply(classify_theme)

            result['priority'] = result.apply(
                calculate_priority_score,
                axis=1
            )

            st.session_state['search_results'] = result

if not st.session_state['search_results'].empty:

    df_result = st.session_state['search_results']

    st.markdown("<div class='filter-box'>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)

    with f1:

        sentiment_filter = st.multiselect(
            "Тональность",
            df_result['sentiment'].unique(),
            default=df_result['sentiment'].unique()
        )

    with f2:

        theme_filter = st.multiselect(
            "Темы",
            df_result['theme'].unique(),
            default=df_result['theme'].unique()
        )

    with f3:

        priority_filter = st.slider(
            "Минимальный приоритет",
            1.0,
            5.0,
            1.0
        )

    st.markdown("</div>", unsafe_allow_html=True)

    filtered = df_result[
        (df_result['sentiment'].isin(sentiment_filter))
        &
        (df_result['theme'].isin(theme_filter))
        &
        (df_result['priority'] >= priority_filter)
    ]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Комментариев", len(filtered))

    with c2:
        st.metric(
            "Негатив",
            len(
                filtered[
                    filtered['sentiment'] == 'Негатив'
                ]
            )
        )

    with c3:

        top_theme = (
            filtered['theme'].mode()[0]
            if not filtered.empty
            else "-"
        )

        st.metric("Главная тема", top_theme)

    with c4:

        avg_priority = round(
            filtered['priority'].mean(),
            1
        )

        st.metric(
            "Средний приоритет",
            avg_priority
        )

    tab1, tab2, tab3 = st.tabs([
        "📊 Аналитика",
        "💬 Комментарии",
        "🧠 Инсайты"
    ])

    with tab1:

        g1, g2 = st.columns(2)

        with g1:

            fig = px.pie(
                filtered,
                names='sentiment',
                hole=0.45,
                color='sentiment',
                color_discrete_map={
                    'Позитив':'#FF8FB1',
                    'Негатив':'#E75480',
                    'Нейтрально':'#FFB7CE'
                }
            )

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with g2:

            text = " ".join(
                filtered['comment_text']
                .astype(str)
            )

            wc = WordCloud(
                width=800,
                height=400,
                background_color=None,
                mode="RGBA",
                colormap="RdPu"
            ).generate(text)

            fig_wc, ax = plt.subplots()

            ax.imshow(wc)

            ax.axis("off")

            fig_wc.patch.set_alpha(0)

            st.pyplot(fig_wc)

        theme_stats = (
            filtered['theme']
            .value_counts()
            .reset_index()
        )

        theme_stats.columns = ['theme', 'count']

        fig2 = px.bar(
            theme_stats,
            x='theme',
            y='count',
            color='theme',
            color_discrete_sequence=[
                "#E75480",
                "#FF8FB1",
                "#FFB7CE",
                "#FFC2D1"
            ]
        )

        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    with tab2:

        table_df = filtered.copy()

        table_df["comment_date"] = pd.to_datetime(
            table_df["comment_date"],
            errors="coerce"
        )

        table_df["comment_length"] = (
            table_df["comment_text"]
            .astype(str)
            .apply(len)
        )

        st.markdown("### Фильтрация комментариев")

        fc1, fc2, fc3, fc4 = st.columns(4)

        with fc1:

            sort_option = st.selectbox(
                "Сортировка",
                [
                    "По приоритету",
                    "По новизне",
                    "По длине"
                ]
            )

        with fc2:

            min_length = st.slider(
                "Мин. длина",
                0,
                500,
                0
            )

        with fc3:

            max_length = st.slider(
                "Макс. длина",
                0,
                1000,
                1000
            )

        with fc4:

            only_questions = st.checkbox(
                "Только вопросы"
            )

        table_df = table_df[
            (table_df["comment_length"] >= min_length)
            &
            (table_df["comment_length"] <= max_length)
        ]

        if only_questions:

            table_df = table_df[
                table_df["comment_text"]
                .astype(str)
                .str.contains(r"\?")
            ]

        if sort_option == "По приоритету":

            table_df = table_df.sort_values(
                by="priority",
                ascending=False
            )

        elif sort_option == "По новизне":

            table_df = table_df.sort_values(
                by="comment_date",
                ascending=False
            )

        elif sort_option == "По длине":

            table_df = table_df.sort_values(
                by="comment_length",
                ascending=False
            )

        show_df = table_df[[
            "comment_text",
            "comment_author",
            "comment_date",
            "sentiment",
            "theme",
            "priority",
            "comment_length",
            "post_url"
        ]]

        st.dataframe(
            show_df,
            use_container_width=True,
            hide_index=True,
            height=650,
            column_config={

                "comment_text": st.column_config.TextColumn(
                    "Комментарий",
                    width="large"
                ),

                "comment_author": st.column_config.TextColumn(
                    "Автор"
                ),

                "comment_date": st.column_config.DatetimeColumn(
                    "Дата"
                ),

                "sentiment": st.column_config.TextColumn(
                    "Тональность"
                ),

                "theme": st.column_config.TextColumn(
                    "Тема"
                ),

                "priority": st.column_config.ProgressColumn(
                    "Приоритет",
                    min_value=1.0,
                    max_value=5.0,
                    format="%.1f"
                ),

                "comment_length": st.column_config.NumberColumn(
                    "Длина"
                ),

                "post_url": st.column_config.LinkColumn(
                    "Пост"
                )
            }
        )

        st.markdown(f"""
        Найдено комментариев: **{len(show_df)}**
        """)

    with tab3:

        insights = generate_insights(filtered)

        for i in insights:

            st.markdown(f"""
            <div class='insight-box'>
                {i}
            </div>
            """, unsafe_allow_html=True)

    csv = filtered.to_csv(
        index=False
    ).encode('utf-8-sig')

    st.download_button(
        "📥 Скачать отчет CSV",
        csv,
        "report.csv",
        "text/csv"
    )
