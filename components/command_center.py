import pandas as pd
import streamlit as st
import os
from os.path import join
from pandas.api.types import is_numeric_dtype
from scripts.constants import PERCENT_RANK, TIER, MARKET_LEVELS, VARIABLE_CORRELATION_THRESHOLD, AUDIENCE_BUILDER_DATASETS
from scripts.matched_market import calculate_tier, MatchedMarketScoring

def render_command_center():
    """
    Render the Command Center tab in the Streamlit app.
    """

    # Set the current directory
    cd = os.getcwd()

    # Initialize empty dataframes
    kpi_df, client_df, agg_kpi_df, audience_df = None, None, None, None
    client_columns, audience_columns = [], []

    # Expander for Market Level Selection
    with st.expander(label="**Market Level Selection**", expanded=True):
        market_level = st.selectbox(
            label="**Select a Market Level**",
            options=MARKET_LEVELS,
            help="Choose a Market Level for Market Scoring & Market Matching"
        )

    # Expander for Target data selection
    with st.expander(label="**Target Audience**", expanded=True):

        audience_file = f"{market_level.replace(' ', '_').lower()}_audience.csv"
        audience_path = join(cd, 'data', 'audience', audience_file)
        audience_df = pd.read_csv(audience_path)
        st.dataframe(audience_df)

        pass # TODO

    # Expander for Data Upload
    with st.expander(label="**Data Uploader**", expanded=True):
        pass #TODO

    # Expander for KPI Selection and Data Granularity
    with st.expander(label="**KPI Selection & Data Granularity**", expanded=True):
        pass #TODO

    # Expander for Additional Data Sources
    with st.expander(label="**Incorporating Additional Data Sources**", expanded=False):
        if kpi_df is not None and market_level is not None:
            pass #TODO

    # Run market ranking and matching if data is ready

    ## TODO : code for running the process
   
    # Footer markdown with contact info
    st.markdown("***")
    st.markdown(
        "If you have any questions about the KPI or Audience data required, please reach out to Media Analytics "
        "team at MediaAnalytics@mediamonks.com"
    )