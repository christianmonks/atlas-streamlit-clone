import pandas as pd
import streamlit as st
import os
from os.path import join
from pandas.api.types import is_numeric_dtype
from scripts.constants import (
    PERCENT_RANK, 
    TIER, 
    MARKET_LEVELS, 
    VARIABLE_CORRELATION_THRESHOLD, 
    AUDIENCE_BUILDER_DATASETS, 
    MARKETS
)
from scripts.matched_market import calculate_tier, MatchedMarketScoring

MARKET_COLUMN = 'Market'

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

        # TODO: Additional functionalities can be added here.

    # Expander for Data Upload
    with st.expander(label="**Data Uploader**", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            # KPI data uploader
            st.write("")
            uploaded_file_kpi = st.file_uploader("**Upload Client KPI Data**")
            if uploaded_file_kpi is not None:
                # Load and display KPI data if uploaded
                kpi_df = pd.read_csv(uploaded_file_kpi)
                kpi_column_exists = any("kpi" in v.lower() for v in kpi_df.columns)
                if len(kpi_df) > 0 and kpi_column_exists:
                    st.success(
                        f"Successfully Loaded KPI Data. Total of {len(kpi_df)} Records. A Snapshot Is Provided Below."
                    )
                    st.dataframe(kpi_df, hide_index=True)
                else:
                    st.error("Please load a compatible dataset with a defined KPI column.")
            else:
                st.error("No Client KPI Data Uploaded", icon="🚨")

        with col2:
            st.write("")
            uploaded_file_client = st.file_uploader("**Upload Optional Client Specific Data**")
            if uploaded_file_client is not None:
                client_df = pd.read_csv(uploaded_file_client)
                if len(client_df) > 0:
                    st.success(
                        f"Successfully Loaded Client Data. Total of {len(client_df)} Records. A Snapshot Is Provided Below."
                    )
                    st.dataframe(client_df, hide_index=True)
            else:
                st.error("No Additional Client Data Uploaded", icon="🚨")

    # Expander for Data Granularity
    with st.expander(label="**KPI Selection and Data Granularity**", expanded=True):
        # Ensure KPI data is uploaded before proceeding
        if kpi_df is None:
            st.error("Please return to the previous expander and upload KPI data.", icon="🚨")
        else:
            kpi_columns = {
                i.replace("KPI", "").title().replace("_", " ").strip(): i
                for i in kpi_df.columns if "kpi" in i.lower()
            }
            kpi_column = st.selectbox(
                label="**Select a KPI Column to Rank Markets By**",
                options=list(kpi_columns.keys()),
                help="Choose a KPI column from the available options. Markets will be grouped into tiers based on the selected KPI."
            )
            

            rename_dict = {
                market_level: MARKET_COLUMN,
                kpi_columns.get(kpi_column): kpi_column,
            }

            # Rename columns in client dataframe if it exists
            if client_df is not None:
                client_columns = {i: i.title().replace("_", " ") for i in client_df.columns if market_level.lower() not in i.lower()}
                client_df.rename(columns=lambda col: rename_dict.get(col, col), inplace=True)
                client_df.rename(columns=lambda col: client_columns.get(col, col), inplace=True)
                client_columns = list(client_columns.values())
            
            # Rename columns in KPI dataframe
            kpi_df.rename(columns=lambda col: rename_dict.get(col, col), inplace=True)

            print(f'esto es lo que queremos ver kpi_columns: {kpi_columns}')
            date_columns = [k for k in kpi_df.columns if k not in [MARKET_COLUMN] + list(kpi_columns.keys())]

            # Check if market level exists in KPI data
            if MARKET_COLUMN in kpi_df.columns:
                if is_numeric_dtype(kpi_df[kpi_column]):
                    num_tiers = st.number_input(
                        min_value=1,
                        max_value=10,
                        value=4,
                        label="**Number of KPI Tiers**",
                        help="For numeric KPI columns, choose the number of tiers to group markets by performance."
                    )
                    agg_kpi_df = kpi_df.groupby(MARKET_COLUMN)[[kpi_column]].sum().reset_index()
                    agg_kpi_df[PERCENT_RANK] = agg_kpi_df[kpi_column].rank(pct=True)
                    agg_kpi_df[TIER] = agg_kpi_df[PERCENT_RANK].apply(lambda x: calculate_tier(x, num_tiers))
                else:
                    # Non-numeric KPI - Copy as is
                    agg_kpi_df = kpi_df.copy()
                    agg_kpi_df[TIER] = agg_kpi_df[kpi_column]

                print(f'esto es lo que queremos ver date_columns: {date_columns}')

                if date_columns:
                    # Select a date column if available
                    date_column = st.selectbox(
                        label="**Select a Date Column**",
                        options=['Daily', 'Weekly', 'Aggregated'],
                        help="Select the date column representing the time period of the KPI."
                    )
                else:
                    date_column = None
            else:
                st.error("Confirm Market Level", icon="🚨")


        
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