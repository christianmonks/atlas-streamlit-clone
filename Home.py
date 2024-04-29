import json
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
from plotly.offline import init_notebook_mode, iplot

from scripts.utils import *
from scripts.constants import *
from scripts.matched_market import MatchedMarketScoring, \
    calculate_tier, generate_dma_data

# Page Config
st.set_page_config(
    page_title=None, page_icon=None, layout="wide",
    initial_sidebar_state="auto", menu_items=None
)

# Add Title
image = add_image()
col1, col2, col3 = st.columns([1, .5, 1])
with col1:
    st.write("")
with col2:
    st.image(image, width=450, use_column_width=True)
with col3:
    st.write("")

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.write("")
with col2:
    st.markdown(
        "<h3 style='text-align: center;'> 👋  Welcome to Matched Market Testing Suite </h3>", unsafe_allow_html=True
    )
with col3:
    st.write("")

# Current working directory
cd = os.getcwd()
world_country_df = pd.read_csv(
    join(cd, 'data', 'mmt_world_country.csv')
)

# initialize session state
if 'mm' not in st.session_state:
    st.session_state.mm = None

col1, col2, col3 = st.columns([1, 7, 1])
with col2:
    st.write("")
    st.write("""
    **The Matched Market Testing Suite (MMT) is designed to craft potent market testing strategies tailored for brands and advertisers. 
    Utilizing cutting-edge AI and machine learning technologies, it discerns the interplay between business KPIs and various demographic, 
    economic, and media variables. By identifying key factors and quantifying their impact on business KPIs,
    MMT empowers informed decision-making.** 
    """)
    tab1, tab2, tab3, tab4 = st.tabs([
        "**Product Overview**", "**Matched Market Command Center**",
        "**Market Rankings & Insights**", "**Matched Markets**"
    ])

# tab1: instructions
with tab1:
    with st.expander('**Features of the Matched Market Testing Suite**', expanded=True):
        st.markdown("* **Market Prioritization Tool**: Leveraging factor weights returned by the machine learning model, it provides a \
              market ranking that can used to prioritize markets for media testing and market expansion purposes.")
        st.markdown("* **Matched Markets Tool**: Leveraging market ranking and factor weights, it provides matched market pairs that can \
              be used for media testing and uplift modeling.")
    with st.expander('**Enhanced Capabilities of the Matched Market Testing Suite**', expanded=False):
        st.markdown("* **Diverse Market Analysis**: MMT facilitates comprehensive assessments at both Country and US DMA levels.")
        st.markdown("* **Adaptable Metrics**: MMT accommodates various KPIs, encompassing numerical indicators like sales volume,"
                    " as well as qualitative measures such as Brand Equity levels (Low/Medium/High).")
        st.markdown("* **Versatile KPI Utilization**: MMT empowers market evaluation and comparison through the utilization of diverse business KPIs.")
        st.markdown("* **Broad Data Integration**: MMT seamlessly integrates 1st and 3rd-party data from brands and advertisers, enriching analyses with robust insights.")
    with st.expander('**Illustrative Examples of the Matched Market Testing Suite**', expanded=False):
        st.markdown("* **European Fashion Retailer**: Strategic Market Assessment for a European online fashion retailer (Primary Business KPI: Brand Equity)")
        st.markdown("* **Global Investment Fund**: Market Prioritization for a Global Investment Fund (Primary Business KPI: Not specified)")
        st.markdown("* **US Fashion Retailer**: Tailored Market Analysis for a Fashion Brand in U.S. Markets (Primary Business KPI: Sales Volume)")

# tab2: Matched Market Command Center
with tab2:
    kpi_df, audience_df, agg_kpi_df = None, None, None
    with st.expander(label="**Data Uploader**", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("")
            uploaded_file_kpi = st.file_uploader("**Upload KPI data**")
            st.info(
                """👆 To protect your data and privacy while demoing this prototype, please upload a .csv file first. 
                Sample to try: [kpi_data_example.csv](https://drive.google.com/file/d/1n6gayg5VcWRCtJ5qVIfRwjc08WJmed3y/view?usp=sharing)."""
            )
            if uploaded_file_kpi is not None:
                # Can be used wherever a "file-like" object is accepted:
                kpi_df = pd.read_csv(uploaded_file_kpi)
                kpi_column_exists = any('kpi' in v.lower() for v in kpi_df.columns)
                if (len(kpi_df) > 0) & kpi_column_exists:
                    st.success(
                       f"Successfully Loaded KPI Data. Total of {len(kpi_df)} Records. A Snapshot Is Provided Below."
                    )
                    st.dataframe(kpi_df, hide_index=True)
                else:
                    st.error(f"Please Load a Compatible Dataset with a Defined KPI Column.")
        with col2:
            st.write("")
            uploaded_file_audience = st.file_uploader("**Upload Audience Data**")
            st.info(
                """👆 To protect your data and privacy while demoing this prototype, please upload a .csv file first. 
                Sample to try: [audience_data_example.csv](https://drive.google.com/file/d/1e57TDVk4LyjeLBSCHZ5I6eCeS3QX46FR/view?usp=sharing)."""
            )
            if uploaded_file_audience is not None:
                # Can be used wherever a "file-like" object is accepted:
                audience_df = pd.read_csv(uploaded_file_audience)
                audience_column_exists = any('audience' in v.lower() for v in audience_df.columns)
                if (len(audience_df) > 0) & audience_column_exists:
                    st.success(
                       f"Successfully Loaded Audience Data. Total of {len(audience_df)} Records. A Snapshot Is Provided Below."
                    )
                    st.dataframe(audience_df, hide_index=True)
                else:
                    st.error(f"Please Load a Compatible Dataset with a Defined Audience Column.")

    with st.expander(label='**KPI Selection**'):
        col1, col2 = st.columns([1, 1])
        if (kpi_df is None) | (audience_df is None):
            st.error('Please Return to the Previous Expander and Upload Audience and KPI Data', icon="🚨")
        else:
            with col1:
                agg_kpi_df = None
                kpi_columns = [i for i in list(kpi_df) if 'kpi' in i.lower()]
                kpi_column = st.selectbox(
                    label="**Select a KPI Column to Rank Markets By**",
                    options=kpi_columns,
                    help="Choose a Key Performance Indicator (KPI) column from the available options. "
                         "Markets will be grouped into tiers based on the performance of the selected KPI column."
                )

                audience_columns = [i for i in list(audience_df) if 'audience' in i.lower()]
                market_level = 'DMA' if len([c for c in list(kpi_df) if 'dma' in c.lower()]) > 0 else 'World'
                market_column = DMA_CODE if market_level == 'DMA' else COUNTRY_CODE

                rename_dict = {
                    'DMA_CODE': DMA_CODE,
                    'COUNTRY_CODE': COUNTRY_CODE,
                    'COUNTRY_NAME': COUNTRY_NAME
                }

                kpi_df = kpi_df.rename(columns=lambda col: rename_dict.get(col, col))
                audience_df = audience_df.rename(columns=lambda col: rename_dict.get(col, col))
                world_country_df = world_country_df.rename(columns=lambda col: rename_dict.get(col, col))

                if is_numeric_dtype(kpi_df[kpi_column]):
                    with col2:
                        num_tiers = st.number_input(
                            min_value=1, max_value=10, value=4, label='**Number of KPI Tiers**',
                            help="If your KPI column is numeric, choose the number of tiers into which "
                                 "markets can be grouped based on its performance."
                        )
                        agg_kpi_df = kpi_df.groupby(market_column)[[kpi_column]].sum().reset_index()
                        agg_kpi_df[PERCENT_RANK] = agg_kpi_df[kpi_column].rank(pct=True)
                        agg_kpi_df[TIER] = agg_kpi_df[PERCENT_RANK].apply(lambda x: calculate_tier(x, num_tiers))
                else:
                    agg_kpi_df = kpi_df
                    agg_kpi_df[TIER] = agg_kpi_df[kpi_column]

    with st.expander(label="**Incorporating Additional Data Sources**"):
        if (kpi_df is None) | (audience_df is None):
            st.error('Please Return to the Previous Expander and Upload Audience and KPI Data', icon="🚨")
        else:
            if market_level == 'DMA':
                dma_path = join(cd, 'data', 'census_dma')
                dma_files = [
                    f for f in os.listdir(dma_path) if \
                    os.path.isfile(os.path.join(dma_path, f)) and f != '.DS_Store' and f in INCLUDE_DMA
                ]

                dma_dfs = {
                    v.replace('.csv', '').replace('_', ' ').title().replace('Dma', 'DMA').replace(
                        'Cpm', 'CPM').replace('Gdp', 'GDP'): pd.read_csv(
                        join(dma_path, v), usecols=DMA_COLUMNS[v])
                    for i, v in enumerate(dma_files)
                }
                dma_included = st.multiselect(
                    label='**Select DMA Census Data to Include**',
                    options=list(dma_dfs.keys()),
                    default=DEFAULT_DMA,
                    help="Choose additional DMA census datasets to include alongside"
                         " your audience dataset for creating a Market Score."
                )
                dma_data = generate_dma_data(dma_dfs, dma_included)
                df = audience_df.merge(dma_data, on=market_column, how='inner').merge(
                    agg_kpi_df, on=market_column, how='inner'
                )

            else:
                df = audience_df.merge(world_country_df, on=market_column, how='inner').merge(
                    agg_kpi_df, on=market_column, how='inner'
                )

            null_percentage = (df.isnull().sum() / len(df)) * 100
            columns_to_drop = null_percentage[null_percentage > 50].index
            df = df.drop(columns=columns_to_drop)

            cov_columns = [c for c in list(dma_data) if (c not in [DMA_NAME, DMA_CODE]) and (c in list(df))] if market_level == 'DMA' else \
                [c for c in list(world_country_df) if (c not in [COUNTRY_CODE, COUNTRY_NAME]) and (c in list(df))]

            cov_columns = {
                c: c.title().replace('_', ' ') for c in cov_columns
            }

            df = df.rename(columns=cov_columns)
            cov_columns = [v for k, v in cov_columns.items()]
            included_cov = st.multiselect(
                label='**Select Demographic Factors to Include or Exclude**',
                options=cov_columns,
                default=cov_columns,
                help="Choose specific demographic factors to include or exclude from your analysis."
            )

            df_columns = [COUNTRY_CODE, COUNTRY_NAME] + included_cov + audience_columns + [kpi_column, TIER] if \
                market_level != 'DMA' else [DMA_CODE, DMA_NAME] + included_cov + audience_columns + [kpi_column, TIER]
            df = df[df_columns]
            if st.checkbox("View Merged KPI, Audiences and Market Data"):
                st.dataframe(df, hide_index=True)
            st.success("Successfully Merged KPI, Audiences, and Market data. Review the Merged data below.")

    if agg_kpi_df is not None:
        bt_run_market_ranking = st.button(label="**Confirm and Run Market Ranking 🏃‍➡**")
        if bt_run_market_ranking:
            with st.spinner(text="Running ML Model to Calculate Market Scoring & Matching..."):
                mm = MatchedMarketScoring(
                    df=df,
                    audience_columns=audience_columns,
                    display_columns=[DMA_CODE, DMA_NAME] if market_level == 'DMA'\
                        else [COUNTRY_CODE, COUNTRY_NAME],
                    covariate_columns=cov_columns,
                    market_column=market_column
                )
                st.session_state.mm = mm
            st.success("Successfully Ran Market Scoring & Matching ✅")
    st.markdown("***")
    st.markdown(
        "If you have any questions about the KPI or Audience data required, please reach out to Media Analytics " \
        "team at MediaAnalytics@mediamonks.com"
    )

# tab3: Market Rankings & Insights
with tab3:
    if st.session_state.mm and kpi_df is not None and audience_df is not None:
        mm = st.session_state.mm
        display_df1 = mm.fi[[FEATURE, WEIGHT]].head(10)
        display_df1[WEIGHT] = display_df1[WEIGHT]/display_df1[WEIGHT].sum()
        display_df2 = mm.ranking_df.reset_index(drop=True)
        col1, col2, col3, col4 = st.columns([.25, .25, .5,.25], gap='small')
        with col2:
            top_n = st.number_input(
                label='**Top Markets Per Tier**',
                min_value=1,
                max_value=10,
                value=4,
                help="Choose the number of top markets you want displayed per tier."
            )
        with col3:
            tier_filter = st.multiselect(
                label='**Tier Filter**',
                options=set(mm.ranking_df[TIER]),
                default=set(mm.ranking_df[TIER]),
                help="Choose which tiers to include or exclude from the display."
            )

        st.write("")
        if market_level == 'DMA':
            with st.expander("**DMA Market Score Heat Map**", expanded=True):
                with open("dma.json") as geofile:
                    source_geo = json.load(geofile)
                perf_df = mm.ranking_df.reset_index(drop=True)
                perf_df = perf_df[perf_df[TIER].isin(tier_filter)]
                fig_heat = px.choropleth(
                    perf_df,
                    geojson=source_geo,
                    locations=DMA_CODE,
                    color='Score',
                    color_continuous_scale="YlOrRd",
                    range_color=(0, perf_df['Score'].max()),
                    scope="usa",
                    labels={'Score':'Market Score'},
                    hover_data=[DMA_NAME],
                    height=1000,
                    width=1200,
                    title=" <span style='font-size:30px;color:black;'>DMA Market Score Heat Map </span>",
                )
                fig_heat.update_layout(
                    title_x=0.375,
                    title_y=0.9
                )
                st.plotly_chart(
                    fig_heat, theme='streamlit', use_container_width=True
                )


        with st.expander(
                f"**Socioeconomic Weights for {kpi_column.replace('KPI', '').title().replace('_', ' ')} & Market Rankings**", expanded=True
        ):
            col1, col2, col3 = st.columns([.2, 1.5, 1.75])
            with col2:
                fig_feature_weights = px.pie(
                    display_df1,
                    values=WEIGHT,
                    names=FEATURE,
                    color=FEATURE,
                    title=f"Socioeconomic Feature Weights for {kpi_column.replace('KPI', '').title().replace('_', ' ')} Tiers",
                    width=600,
                    height=500,
                )
                st.plotly_chart(
                    fig_feature_weights, theme='streamlit', use_container_width=False
                )

            with col3:
                display_df2 = display_df2[
                    (display_df2[TIER].isin(tier_filter)) &
                    (display_df2['Tier Rank'] <= top_n)
                ].sort_values(by=[TIER, 'Score'], ascending=[False, True])
                graph_col = DMA_NAME if market_level == 'DMA' else COUNTRY_NAME
                fig_ranking = px.bar(
                    display_df2,
                    x=SCORE,
                    y=graph_col,
                    orientation='h',  # horizontal bar chart
                    labels={graph_col: 'Market Rank', SCORE: 'Market Score'},
                    title=f"Top {top_n} Markets Per Tier Based on {kpi_column.replace('KPI', '').title().replace('_', ' ')} Score",
                    width=600,
                    height=500,
                    color=TIER
                )
                st.plotly_chart(fig_ranking, theme='streamlit')

            st.write("")
            col1, col2, col3 = st.columns([1, 1.5, 1])
            with col2:
                st.markdown(
                    f"<h5 style='text-align: center; color: black;'>Top {top_n} "
                    f"Markets Per Tier Based on {kpi_column.replace('KPI', '').title().replace('_', ' ')} Score</h5>",
                    unsafe_allow_html=True
                )
                st.dataframe(
                    display_df2[[
                        TIER, market_column, graph_col,
                        'Score', 'Tier Rank'
                    ]].sort_values(
                        by=[TIER, 'Score'], ascending=[True, False]
                    ),
                    hide_index=True, use_container_width=True
                )

    else:
        st.error('Please Return to the Previous Tab and Upload Audience and KPI Data', icon="🚨")

# tab4: Matched Markets
with tab4:
    if st.session_state.mm and kpi_df is not None and audience_df is not None:
        mm = st.session_state.mm
        mm_df = mm.similar_markets

        col1, col2, col3, col4 = st.columns([1, 1, 1, .6], gap='small')
        with col1:
            tier_filter = st.multiselect(
                label="**Select Tiers for Identifying Similar Matched Markets***",
                options=list(set(mm_df['Tier'])),
                default=['Tier 1'],
                help="Choose the tiers you want to use for identifying similar matched markets."
            )
            tier_mask = mm_df['Tier'].isin(tier_filter)

        with col2:
            market_removal = st.multiselect(
                label="**Select Markets to Exclude from Test**",
                options=list(set(mm_df[tier_mask]['Test Market Name'])),
                help="Choose markets that you want to exclude from matched market pairing."
            )
            removal_mask = (mm_df['Test Market Name'].isin(market_removal))

        with col3:
            specific_markets = st.multiselect(
                label="**Select Specific Test Markets**",
                options=list(set(mm_df[tier_mask & ~removal_mask]['Test Market Name'])),
                help="Choose specific test markets to include for matched market pairing."
            )
            spec_mask = mm_df['Test Market Name'].isin(specific_markets) if \
                specific_markets else ~mm_df['Test Market Name'].isin([])

        with col4:
            num_pairs = st.number_input(
                '**Number of Market Pairs**',
                min_value=1,
                max_value=10,
                value=5 if len(specific_markets) == 0 else len(specific_markets),
                help="Specify the number of test and control market pairs to generate."
            )

        mm_df = mm_df[tier_mask & ~removal_mask & spec_mask]
        col1, col2, col3 = st.columns([1, 4, 1], gap='medium')

        with col2:
            if len(tier_filter) > 0:
                counter = 0
                utilized_markets = []
                matched_df = pd.DataFrame()
                max_counter = num_pairs*len(tier_filter) if len(specific_markets) == 0 else \
                    len(specific_markets)*num_pairs
                while counter < max_counter:

                    control_market_mask = (~mm_df['Control Market Name'].isin(utilized_markets))
                    test_market_mask = (~mm_df['Test Market Name'].isin(utilized_markets))

                    mm_df1 = mm_df[control_market_mask & test_market_mask].sort_values(
                        by='Similarity Index',ascending=True
                    )
                    mm_df1['Rank'] = mm_df1.groupby(['Tier']).cumcount()+1
                    matched_df = pd.concat([
                        matched_df,
                        mm_df1[mm_df1['Rank'] == 1]
                    ], axis=0)
                    utilized_markets.extend(
                      [c for c in mm_df1[mm_df1['Rank'] == 1]['Control Market Name']] + \
                      [c for c in mm_df1[mm_df1['Rank'] == 1]['Test Market Name']]
                    )
                    counter += len(tier_filter) if len(specific_markets) == 0 else len(specific_markets)

                st.write("")
                st.write("")
                st.markdown(
                    f"<h5 style='text-align: center; color: black;'>Matched Markets based on Similarity Index</h5>",
                    unsafe_allow_html=True
                )

                matched_df = matched_df.drop('Rank', axis=1)
                st.dataframe(
                    matched_df.sort_values(
                        by='Tier',
                        ascending=True),
                        hide_index=True,
                        use_container_width=True
                )
    else:
        st.error('Please Return to the Previous Tab and Upload Audience and KPI Data', icon="🚨")
