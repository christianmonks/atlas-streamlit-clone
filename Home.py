# landing page
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# load data
# load & process standard data
# separate dma and country datasets
us_dma_df = pd.read_csv('./data/mmt_standard_data.csv')
us_dma_df = us_dma_df[(us_dma_df.MEDIAN_HOUSEHOLD_INCOME > 0) & (us_dma_df.MEDIAN_HOUSING_VALUE > 0) \
                     & (us_dma_df.GDP_PER_CAPITA > 0)].reset_index(drop=True)

# initialize session state
if 'ranking_df' not in st.session_state:
    st.session_state.ranking_df = None

# title
st.markdown("<h1 style='text-align: left;'>Welcome to Matched Markets Testing Beta 👋</h1>", unsafe_allow_html=True)
st.header("")

# tabs
tab1, tab2, tab3, tab4 = st.tabs(["Instructions", "Data Uploader", "Market Prioritization", "Matched Markets"])

# tab1: instructions
with tab1:
    st.write(f"""
    The Matched Market Testing suite is built to help clients create effective market testing strategy. Leveraging 
    modern AI and machine learning technologies, it examines the relationship between business KPIs and demographic, 
    economic and media factors, and returns the leading factors and how much each factor contributes the business KPI.
    It offers two core functionalities, as described below. 
    
    * Marketing Prioritization Tool
    * Matched Markets Tool
    
    It has the following features to support different use cases.
    1. Flexible KPI
    2. Multiple KPI support
    3. Multiple data sources 
    
    Quick Start
    * KPI Template
    * Audience Template 
    """)

# tab 2: data uploader
with tab2:
    st.subheader("Step 1: Upload your data")
    # kpi data
    kpi_df = None
    uploaded_file_kpi = st.file_uploader("Upload your KPI data")
    if uploaded_file_kpi is not None:
        # Can be used wherever a "file-like" object is accepted:
        kpi_df = pd.read_csv(uploaded_file_kpi)
        if len(kpi_df) > 0:
            st.success(f"Successfully loaded KPI data. Rows = {len(kpi_df)}. A snapshot is provided below. ")
            st.dataframe(kpi_df.head(5), hide_index=True)

    # audience data
    audience_df = None
    uploaded_file_audience = st.file_uploader("Upload your audience data")
    if uploaded_file_audience is not None:
        # Can be used wherever a "file-like" object is accepted:
        audience_df = pd.read_csv(uploaded_file_audience)
        if len(audience_df) > 0:
            st.success(f"Successfully loaded audience data. Rows = {len(audience_df)}. A snapshot is provided below.")
            st.dataframe(audience_df, hide_index=True)

# tab 3: market prioritization
with tab3:
    # kpi selection
    agg_kpi_df = None
    if kpi_df is not None and audience_df is not None:
        kpi_columns = [i for i in list(kpi_df) if 'KPI' in i]
        audience_columns = [i for i in list(audience_df) if 'AUDIENCE' in i]
        kpi_column = st.selectbox(label="Select a business KPI", options=kpi_columns)
        market_column = list(kpi_df)[0]
        market_level = market_column.split('_')[0]

        # process kpi data
        # kpi_df['WEEK'] = kpi_df['WEEK'].astype('datetime64[ns]')
        agg_kpi_df = kpi_df.groupby(market_column)[[kpi_column]].sum().reset_index()
        agg_kpi_df['pct_rank'] = agg_kpi_df[kpi_column].rank(pct=True)
        agg_kpi_df['KPI_TIER'] = np.where(agg_kpi_df.pct_rank <= 0.25, 'Tier 4', \
                                          np.where(agg_kpi_df.pct_rank <= 0.5, 'Tier 3', \
                                                   np.where(agg_kpi_df.pct_rank <= 0.75, 'Tier 2', 'Tier 1')))
        agg_kpi_df = agg_kpi_df[[market_column, kpi_column, 'KPI_TIER']]

        # set data based on market level
        standard_df = us_dma_df if market_level == 'DMA' else None
        standard_columns = ['DMA_CODE', 'DMA_NAME', 'UNIVERSE', 'GDP_PER_CAPITA', 'MEDIAN_HOUSEHOLD_INCOME',
                            'MEDIAN_HOUSING_VALUE', 'CPM'] if market_level == 'DMA' else None

    if agg_kpi_df is not None:
        bt_run_market_ranking = st.button(label="Confirm and Run")
        if bt_run_market_ranking:
            df = standard_df[standard_columns].merge(audience_df, on=market_column, how='inner') \
                            .merge(agg_kpi_df, on=market_column, how='inner')
            st.dataframe(df, hide_index=True)
            st.success("Successfully merged KPI, audiences and market data")

            # run RF and calculate scores
            with st.spinner(text="Running ML model for factor weights..."):
                target = 'KPI_TIER'
                # assuming first two columns are market code and name
                model_columns = standard_columns[3 : ] + audience_columns
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

                grid_search = GridSearchCV(estimator=model,
                                           scoring='accuracy',
                                           param_grid=param_grid,
                                           cv=5,
                                           n_jobs=-1,
                                           refit=True,
                                           verbose=0)

                grid_search.fit(X, y)

                best_model = grid_search.best_estimator_
                feature_names = X.columns
                feature_imp = best_model.feature_importances_

                fi = pd.DataFrame({'feature': feature_names, 'score': feature_imp}).sort_values(by=['score'], ascending=False)
                fi['cumulative_score'] = fi['score'].cumsum()

                # calculate scores
                score_df = df[model_columns].copy()
                score_df['CPM'] = 1 / score_df['CPM']
                scaler = MinMaxScaler()
                X_norm = scaler.fit_transform(score_df)
                scores = np.matmul(X_norm, feature_imp)

                ranking_df = pd.DataFrame(X_norm, columns=model_columns)
                ranking_df['Score'] = list(scores)

                display_columns = standard_columns[0 : 2] + [kpi_column, 'KPI_TIER']
                ranking_df = pd.concat([df[display_columns], ranking_df], axis=1)
                ranking_df = ranking_df.sort_values(by=['Score'], ascending=False).reset_index(drop=True)

                st.session_state.ranking_df = ranking_df

            st.success("Successfully calculated factor weights")
            st.dataframe(fi, hide_index=True)
            st.dataframe(ranking_df, hide_index=True)

# tab 4: matched marketws
with tab4:
    st.subheader("Test and Control markets")
    ranking_df = st.session_state.ranking_df

    if ranking_df is not None:
        test_markets = []
        control_markets = []
        for t in range(4):
            market_tier = ranking_df[ranking_df.KPI_TIER == f'Tier {t + 1}'].reset_index(drop=True)
            # by default, first row is the test for the tier
            # calculate distance from the first row
            market_dist = market_tier[model_columns]
            for i in range(1, len(market_dist)):
                market_dist.iloc[i] = market_dist.iloc[0] - market_dist.iloc[i]
            market_dist.iloc[0] = 0
            market_dist = market_dist.abs()

            # calculate weighted
            dist_tier = list(np.matmul(market_dist, feature_imp))[1 : ]
            best_match_index = dist_tier.index(min(dist_tier)) + 1

            # save the test and control geos
            test_markets.append(market_tier[market_column][0])
            control_markets.append(market_tier[market_column][best_match_index])

        test = ranking_df[ranking_df.DMA_CODE.isin(test_markets)]
        control = ranking_df[ranking_df.DMA_CODE.isin(control_markets)]
        st.dataframe(test, hide_index=True)
        st.dataframe(control, hide_index=True)