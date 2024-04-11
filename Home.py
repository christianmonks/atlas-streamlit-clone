import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import os

from os.path import join
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from scripts.utils import*

# page config
st.set_page_config(
    page_title=None, page_icon=None, layout="wide",
    initial_sidebar_state="auto", menu_items=None
)

# title
image = add_image()
col1, col2, col3 = st.columns([1,.5,1])
with col1:
    st.write("")
with col2:
    st.image(image, width=450, use_column_width=True)
with col3:
    st.write("")

col1, col2, col3 = st.columns([1,3,1])
with col1:
    st.write("")
with col2:
    st.markdown("<h3 style='text-align: center;'> 👋  Welcome to Matched Market Testing Suite </h3>", unsafe_allow_html=True)
with col3:
    st.write("")

# load & process standard data
cd = os.getcwd()
us_dma_df = pd.read_csv(join(cd, 'data', 'mmt_us_dma.csv'))
world_country_df = pd.read_csv(join(cd, 'data', 'mmt_world_country.csv'))
us_dma_df = us_dma_df[
    (us_dma_df.MEDIAN_HOUSEHOLD_INCOME > 0) & (us_dma_df.MEDIAN_HOUSING_VALUE > 0) \
    & (us_dma_df.GDP_PER_CAPITA > 0)
].reset_index(drop=True)

# constants
standard_columns_us_dma = [
    'DMA_CODE', 'DMA_NAME', 'UNIVERSE',
    'GDP_PER_CAPITA', 'MEDIAN_HOUSEHOLD_INCOME',
    'MEDIAN_HOUSING_VALUE', 'CPM'
]

standard_columns_world_country = [
    'COUNTRY_CODE', 'COUNTRY_NAME',
    'P18+', 'GDP_PER_CAPITA',
    'MEDIAN_HOUSEHOLD_INCOME', 'CPM'
]

# initialize session state
if 'ranking_df' not in st.session_state:
    st.session_state.ranking_df = None
if 'model_columns' not in st.session_state:
    st.session_state.model_columns = None
if 'feature_weights' not in st.session_state:
    st.session_state.feature_weights = None

col1, col2, col3 = st.columns([1,7,1])
with col2:
    st.write("")
    st.write(f"""
    **The Matched Market Testing Suite (MMT) is built to create effective market testing strategy for brands and 
    advertisers. Leveraging modern AI and machine learning technologies, it learns the relationship between business 
    KPIs and demographic, economic and media factors, and determines what the leading factors are and how much each 
    factor contributes to the business KPI.** 
    """)
    tab1, tab2, tab3, tab4 = st.tabs([
        "**Instructions**", "**Data Uploader**", "**Market Ranking**", "**Matched Markets**"
    ])

# tab1: instructions
with tab1:
    with st.expander('**Features of the Matched Market Testing Suite**', expanded=True):
        st.markdown("* **Market Prioritization Tool**: Leveraging factor weights returned by the machine learning model, it provides a \
              market ranking that can used to prioritize markets for media testing and market expansion purposes.")
        st.markdown("* **Matched Markets Tool**: Leveraging market ranking and factor weights, it provides matched market pairs that can \
              be used for media testing and uplift modeling.")
    with st.expander('**Features of the Matched Market Testing Suite**', expanded=False):
        st.markdown("* Multiple market levels: MMT supports analysis at Country and US DMA levels")
        st.markdown("* Flexible KPIs: MMT supports both numeric KPIs (e.g., sales volume) and non-numeric KPIs (e.g., Brand Equity - Low/Medium/High)")
        st.markdown("* Multiple KPIs: MMT supports market ranking and paring using different business KPIs")
        st.markdown("* Multiple data sources: MMT can ingest 1st and 3rd-party data from brands and advertisers for more powerful insights")
    with st.expander('**Case Studies of the Market Testing Suite**', expanded=False):
        st.markdown('1. Market prioritization for an European online fashion retailer (Primary Business KPI: Brand Equity)')
        st.markdown('2. Market prioritization for a global investment fund (Primary Business KPI: Not defined)')
        st.markdown('3. Matched markets design for a fashion brand in the U.S. markets (Primary Business KPI: Sales Amount)')

with tab2:
    col1, col2 = st.columns([1,1])
    with col1:
        st.write("")
        # kpi data
        kpi_df = None
        uploaded_file_kpi = st.file_uploader("**Upload KPI data**")
        st.info(
        """👆 To protect your data and privacy while demoing this prototype, please upload a .csv file first. 
        Sample to try: [kpi_data_example.csv](https://drive.google.com/file/d/1n6gayg5VcWRCtJ5qVIfRwjc08WJmed3y/view?usp=sharing)."""
        )
        if uploaded_file_kpi is not None:
            # Can be used wherever a "file-like" object is accepted:
            kpi_df = pd.read_csv(uploaded_file_kpi)
            if len(kpi_df) > 0:
                st.success(f"Successfully loaded KPI Data. Rows = {len(kpi_df)}. A snapshot is provided below. ")
                st.dataframe(kpi_df, hide_index=True)

    with col2:
        st.write("")
        # audience data
        audience_df = None
        uploaded_file_audience = st.file_uploader("**Upload Audience Data**")
        st.info(
        """👆 To protect your data and privacy while demoing this prototype, please upload a .csv file first. 
        Sample to try: [audience_data_example.csv](https://drive.google.com/file/d/1e57TDVk4LyjeLBSCHZ5I6eCeS3QX46FR/view?usp=sharing)."""
        )
        if uploaded_file_audience is not None:
            # Can be used wherever a "file-like" object is accepted:
            audience_df = pd.read_csv(uploaded_file_audience)
            if len(audience_df) > 0:
                st.success(f"Successfully loaded audience data. Rows = {len(audience_df)}. A snapshot is provided below.")
                st.dataframe(audience_df, hide_index=True)

    st.write("")
    st.markdown("""
    ***
    """)
    st.markdown(
        "If you have any questions about the KPI or Audience data required, please reach out to Media Analytics" \
        "team at MediaAnalytics@mediamonks.com"
    )

# tab 3: market prioritization
with tab3:
    if (kpi_df is None) | (audience_df is None):
        st.error('Please Return to the Previous Tab and Upload Audience and  KPI Data', icon="🚨")
    else:
        # kpi selection
        agg_kpi_df = None
        if kpi_df is not None and audience_df is not None:
            kpi_columns = [i for i in list(kpi_df) if 'KPI' in i]
            audience_columns = [i for i in list(audience_df) if 'AUDIENCE' in i]
            kpi_column = st.selectbox("**Select a KPI to Rank Markets By**", options=kpi_columns)
            market_column = list(kpi_df)[0]
            market_level = market_column.split('_')[0]

            # process kpi data if numeric
            if is_numeric_dtype(kpi_df[kpi_column]):
                # kpi_df['WEEK'] = kpi_df['WEEK'].astype('datetime64[ns]')
                agg_kpi_df = kpi_df.groupby(market_column)[[kpi_column]].sum().reset_index()
                agg_kpi_df['pct_rank'] = agg_kpi_df[kpi_column].rank(pct=True)
                agg_kpi_df['KPI_TIER'] = np.where(
                    agg_kpi_df.pct_rank <= 0.25, 'Tier 4',
                    np.where(
                        agg_kpi_df.pct_rank <= 0.5, 'Tier 3',
                        np.where(agg_kpi_df.pct_rank <= 0.75, 'Tier 2', 'Tier 1')
                    )
                )
                agg_kpi_df = agg_kpi_df[[market_column, kpi_column, 'KPI_TIER']]
            else:
                agg_kpi_df = kpi_df
                agg_kpi_df['KPI_TIER'] = agg_kpi_df[kpi_column]

            # set data based on market level
            standard_df = us_dma_df if market_level == 'DMA' else world_country_df
            standard_columns = standard_columns_us_dma if market_level == 'DMA' else standard_columns_world_country

        if agg_kpi_df is not None:

            df = standard_df[standard_columns].merge(
                audience_df, on=market_column, how='inner'
            ).merge(
                agg_kpi_df, on=market_column, how='inner'
            )
            df = df.dropna(axis=1)
            st.success("Successfully Merged KPI, Audiences and Market data. Review the merged data below.")
            with st.expander('Audience and Market Merged Dataset', expanded=False):
                st.dataframe(df, hide_index=True)

            bt_run_market_ranking = st.button(
                label="Confirm and Run Market Ranking"
            )

            if bt_run_market_ranking:

                # run RF to calculate weights
                with st.spinner(text="Running ML model to calculate factor weights..."):

                    target = 'KPI_TIER'
                    # assuming first two columns are market code and name
                    model_columns = standard_columns[3 : ] + audience_columns
                    model_columns = [i for i in model_columns if i in list(df)]
                    df = df.dropna(subset=model_columns)
                    X = df[model_columns]
                    y = df[target]

                    param_grid = {
                        'criterion': ['gini', 'entropy', 'log_loss'],
                        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                        'max_features': ['sqrt', None],
                        'n_estimators': [10, 20, 50, 100, 200]
                    }

                    model = RandomForestClassifier(random_state=100)

                    grid_search = GridSearchCV(
                        estimator=model,
                        scoring='accuracy',
                        param_grid=param_grid,
                        cv=5,
                        n_jobs=-1,
                        refit=True,
                        verbose=0
                    )

                    grid_search.fit(X, y)

                    best_model = grid_search.best_estimator_
                    feature_names = X.columns
                    feature_weights = best_model.feature_importances_

                    fi = pd.DataFrame(
                        {'FEATURE': feature_names, 'WEIGHT': feature_weights}
                    ).sort_values(by=['WEIGHT'], ascending=False)
                    fi['CUMULATIVE WEIGHTS'] = fi['WEIGHT'].cumsum()

                    # calculate scores
                    score_df = df[model_columns].copy()
                    score_df['CPM'] = 1 / score_df['CPM']
                    scaler = MinMaxScaler()
                    X_norm = scaler.fit_transform(score_df)
                    scores = np.matmul(X_norm, feature_weights)

                    ranking_df = pd.DataFrame(X_norm, columns=model_columns)
                    ranking_df['SCORE'] = list(scores)

                    display_columns = standard_columns[0 : 2] + [kpi_column, 'KPI_TIER']
                    ranking_df = pd.concat([df[display_columns], ranking_df], axis=1)
                    ranking_df = ranking_df.sort_values(by=['SCORE'], ascending=False).reset_index(drop=True)

                    # update session state
                    st.session_state.ranking_df = ranking_df
                    st.session_state.model_columns = model_columns
                    st.session_state.feature_weights = feature_weights

                st.success("Successfully Calculated Factor weights")
                st.markdown("***")
                st.markdown(f"<h3 style='text-align: left;'> Factor Weights for {kpi_column} </h3>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([3, 3, 1], gap='small')
                display_df1 = fi[['FEATURE', 'WEIGHT']].rename(
                    columns={'FEATURE': 'Feature', 'WEIGHT': 'Weight'})

                with col1:
                     fig_feature_weights = px.pie(
                        display_df1,
                        values ='Weight',
                        names ='Feature',
                        color ='Feature',
                        title =f'Socioeconomic Factor Weights for {kpi_column}',
                        width = 500,
                        height = 500,
                     )
                     st.plotly_chart(fig_feature_weights, theme='streamlit')

                with col2:
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.dataframe(display_df1, hide_index=True)

                st.subheader("Market Rankings")
                ranking_df['Overall Rank'] = ranking_df['SCORE'].rank(ascending=False).astype(int)
                ranking_df['Tier Ranking'] = ranking_df.groupby(['KPI_TIER'])['SCORE'].rank(ascending=False).astype(int)

                col1, col2, col3 = st.columns([3, 3, 1], gap='medium')
                graph_col = [c for c in list(ranking_df) if 'NAME' in c][0]

                with col1:
                    fig1 = px.bar(
                       ranking_df.head(10).sort_values(by='SCORE', ascending = True),
                         x='SCORE',
                         y=graph_col,
                         title=f'Top 10  Markets',
                         orientation='h',  # horizontal bar chart
                         labels={graph_col: 'Market Rank', 'SCORE': 'Market Score'},
                         width = 500,
                         height=500
                     )
                    st.plotly_chart(fig1, theme='streamlit')

                with col2:
                    st.write("")
                    st.write("")
                    st.write("")
                    with st.expander('Market Ranking Table', expanded=False):
                        st.dataframe(ranking_df[[
                            'KPI_TIER', graph_col, 'SCORE', 'Tier Ranking', 'Overall Rank'
                        ]], hide_index=True)

# tab 4: matched markets
with tab4:
    ranking_df = st.session_state.ranking_df
    if ranking_df is None:
        st.error('Please Return to the Previous Tab and Run Market Ranking', icon="🚨")
    else:

        test_markets = []
        control_markets = []

        kpi_tiers = list(ranking_df.KPI_TIER.unique())
        for t in kpi_tiers:
            market_tier = ranking_df[ranking_df.KPI_TIER == t].reset_index(drop=True)
            # by default, first row is the test for the tier
            # calculate distance from the first row
            market_dist = market_tier[st.session_state.model_columns]
            for i in range(1, len(market_dist)):
                market_dist.iloc[i] = market_dist.iloc[0] - market_dist.iloc[i]
            market_dist.iloc[0] = 0
            market_dist = market_dist.abs()

            # calculate weighted
            dist_tier = list(np.matmul(market_dist, st.session_state.feature_weights))[1 : ]
            best_match_index = dist_tier.index(min(dist_tier)) + 1

            # save the test and control geos
            test_markets.append(market_tier[market_column][0])
            control_markets.append(market_tier[market_column][best_match_index])

        mm_display_columns = list(ranking_df)[0 : 4] + ['SCORE']
        test_df = ranking_df[ranking_df[market_column].isin(test_markets)][mm_display_columns]
        control_df = ranking_df[ranking_df[market_column].isin(control_markets)][mm_display_columns]

        test_df.columns = [f'TEST_{i}' if i != 'KPI_TIER' else i for i in list(test_df)]
        control_df.columns = [f'CONTROL_{i}' if i != 'KPI_TIER' else i for i in list(control_df)]

        matched_markets_df = pd.merge(test_df, control_df, on=['KPI_TIER'], how='left').sort_values(by='KPI_TIER')
        mm_display_columns =  ['KPI_TIER'] + \
                              [k for k in list(matched_markets_df) if 'NAME' in k or 'CODE' in k]


        col1, col2, col3 = st.columns([3,1,1])
        with col1:
            st.write("**Matched Markets Based on Similarity Scoring**")
            display_df2 = matched_markets_df[mm_display_columns]
            st.dataframe(display_df2, hide_index=True)
