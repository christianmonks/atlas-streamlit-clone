import pandas as pd
import streamlit as st
import os
from os.path import join
from pandas.api.types import is_numeric_dtype
from scripts.constants import PERCENT_RANK, TIER, MARKET_LEVELS, VARIABLE_CORRELATION_THRESHOLD, AUDIENCE_BUILDER_DATASETS
from scripts.matched_market import calculate_tier, MatchedMarketScoring


def render_command_center():
    """
    Render the Command Center tab in the Streamlit app. This tab handles the uploading of KPI data,
    audience data, market and KPI selection, additional data sources, and running the matched market scoring model.
    """

    # Set the current directory
    cd = os.getcwd()

    # Initialize empty dataframes for KPI and audience data
    kpi_df, client_df, agg_kpi_df, audience_df = None, None, None, None
    client_columns, audience_columns = [], []

    # Expander to upload KPI and optional audience data
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
                if (len(kpi_df) > 0) & kpi_column_exists:
                    st.success(
                         f"Successfully Loaded KPI Data. Total of {len(kpi_df)} Records. A Snapshot Is Provided Below."
                    )
                    st.dataframe(kpi_df, hide_index=True)
                else:
                    st.error(
                        f"Please Load a Compatible Dataset with a Defined KPI Column."
                    )

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

    # Expander for selecting market level and KPI
    with st.expander(label="**Market & KPI Selection**"):
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        market_level, market_code, market_name = None, None, None

        # Ensure KPI data is uploaded before proceeding
        if kpi_df is None:
            st.error("Please Return to the Previous Expander and Upload KPI Data", icon="🚨")
        else:
            with col1:
                # Select market level
                market_level = st.selectbox(
                    label="**Select a Market Level**",
                    options=MARKET_LEVELS,
                    help="Choose a Market Level for Market Scoring & Market Matching",
                )
                market_code = f"{market_level} Code"
                market_name = f"{market_level} Name"

            with col2:
                # Dynamically list KPI columns for selection
                kpi_columns = {
                    i.replace("KPI", "").title().replace("_", " ").strip(): i
                    for i in list(kpi_df) if "kpi" in i.lower()
                }

                # Select KPI column
                kpi_column = st.selectbox(
                    label="**Select a KPI Column to Rank Markets By**",
                    options=list(kpi_columns.keys()),
                    help="Choose a KPI column from the available options. Markets will be grouped into tiers based on the selected KPI.",
                )

            # Create rename dictionary for selected market level and KPI
            rename_dict = {
                market_code.replace(" ", "_").upper(): market_code,
                market_name.replace(" ", "_").upper(): market_name,
                kpi_columns.get(kpi_column): kpi_column,
            }

            # Rename columns in audience dataframe and kpi dataframe
            if client_df is not None:
                client_columns = {i: i.title().replace("_", " ") for i in list(client_df) if market_level.lower() not in i.lower()}
                client_df = client_df.rename(columns=lambda col: rename_dict.get(col, col))
                client_df = client_df.rename(columns=lambda col: client_columns.get(col, col))
                client_columns = [v for k,v in client_columns.items()]

            # Filter non-KPI and non-market columns as date columns
            kpi_df = kpi_df.rename(columns=lambda col: rename_dict.get(col, col))
            date_columns = [k for k in list(kpi_df) if k not in [market_code, market_name] + list(kpi_columns.keys())]

            # Perform additional operations if market code exists in KPI data
            # Numeric KPI - Create tiers based on ranking
            if market_code in list(kpi_df):
                if is_numeric_dtype(kpi_df[kpi_column]):
                    with col3:
                        num_tiers = st.number_input(
                            min_value=1,
                            max_value=10,
                            value=4,
                            label="**Number of KPI Tiers**",
                            help="For numeric KPI columns, choose the number of tiers to group markets by performance.",
                        )
                        agg_kpi_df = kpi_df.groupby(market_code)[[kpi_column]].sum().reset_index()
                        agg_kpi_df[PERCENT_RANK] = agg_kpi_df[kpi_column].rank(pct=True)
                        agg_kpi_df[TIER] = agg_kpi_df[PERCENT_RANK].apply(
                            lambda x: calculate_tier(x, num_tiers)
                        )
                else:
                    # Non-numeric KPI - Copy as is
                    agg_kpi_df = kpi_df.copy()
                    agg_kpi_df[TIER] = agg_kpi_df[kpi_column]

                if len(date_columns) > 0:
                    with col4:
                        # Select a date column if available
                        date_column = st.selectbox(
                            label="**Select a Date Column**",
                            options=date_columns,
                            help="Select the date column representing the time period of the KPI.",
                        )
                else:
                    date_column = None
            else:
                st.error("Confirm Market Level", icon="🚨")

    # Expander for audience builder (DMA level only)
    if market_level == 'DMA':
        with st.expander(label="**Audience Builder**"):
            if st.checkbox("**Create Your Own Audience**"):
                audience_files = {
                    c.title().replace("_", " "):c for c in AUDIENCE_BUILDER_DATASETS
                }
                audience_file  = st.selectbox(
                        label="**Select Your Audience**",
                        options=list(audience_files.keys()),
                        placeholder="No",
                        help="Use US Census Data to Build Your Audience",
                    )
                audience_df = pd.read_csv(f"{join(cd, 'data', 'census_dma', audience_files.get(audience_file))}_dma.csv")
                audience_df = audience_df.loc[:, ~audience_df.columns.str.contains('^Unnamed')]
                audience_columns = {
                    k:k.replace('_', ' ').title() for k in list(audience_df) if k not in [market_code, market_name]
                }
                audience_df = audience_df.rename(columns = audience_columns)
                audience_columns = [k for k in list(audience_df) if k not in  [market_code, market_name, 'Universe']]
                audience_filter  = st.multiselect(
                        label="**Audience Filter**",
                        options=audience_columns,
                        default=audience_columns,
                        placeholder="No",
                        help="Use US Census Data to Build Your Audience",
                    )
                audience_df = audience_df[[market_code]+ audience_filter]
                if len(audience_filter) > 7:
                    st.error("Consider Narrowing Your Audience", icon="🚨")

    # Expander for incorporating additional data sources
    with st.expander(label="**Incorporating Additional Data Sources**"):
        if (kpi_df is None) or (market_code is None) or (market_code not in list(kpi_df)):
            st.error("Please Return to the Previous Expander and Upload Audience and KPI Data", icon="🚨")
        else:
            # Load and process additional data
            additional_data = pd.read_csv(join(cd, 'data', f'mmt_{market_level.lower()}_data.csv'))
            additional_data = additional_data.rename(columns=lambda col: rename_dict.get(col, col))

            if audience_df is not None:
                additional_data = additional_data[[k for k in list(additional_data) if k not in audience_columns]]

            # Start with the list of dataframes
            # Filter out None or empty dataframes
            # Perform merging if there are any valid dataframes
            dfs_to_merge = [additional_data, client_df, audience_df]
            dfs_to_merge = [df for df in dfs_to_merge if df is not None]
            df = agg_kpi_df
            for additional_df in dfs_to_merge:
                df = df.merge(additional_df, on=market_code, how='inner')

            # Drop columns with high null percentage
            null_percentage = (df.isnull().sum() / len(df)) * 100
            columns_to_drop = null_percentage[null_percentage > 10].index
            df = df.drop(columns=columns_to_drop)

            cov_columns = [c for c in additional_data if c not in [market_code, market_name, TIER, kpi_column, 'Percent Rank'] ]
            cov_columns = {c: c.title().replace("_", " ") for c in cov_columns}

            df = df.rename(columns=cov_columns)
            cov_columns = [v for k, v in cov_columns.items()]

            # Identify high correlation covariates if KPI is numeric
            if is_numeric_dtype(kpi_df[kpi_column]):
                corr = df[cov_columns + [kpi_column]].corr()[kpi_column].reset_index()
                corr_vars = [
                    i for i in corr[corr[kpi_column] > VARIABLE_CORRELATION_THRESHOLD]['index'].tolist() \
                    if i != kpi_column and i != 'Universe'
                ]
            else:
                # Default columns for non-numeric KPI
                default_columns = DEFAULT_COLUMNS.get(market_level)
                corr_vars = default_columns

            # Multiselect for demographic factors
            included_cov = st.multiselect(
                label="**Select Demographic Factors to Include or Exclude**",
                options=cov_columns,
                default=corr_vars,
                help="Select demographic factors to include/exclude from the analysis."
            )

            # Final list of columns for analysis
            client_columns = client_columns  if client_columns is not None else []
            df = df[[market_code, market_name] + included_cov + client_columns + audience_columns + [kpi_column, TIER]]

            if st.checkbox("View Merged KPI, Audiences and Market Data"):
                st.dataframe(df, hide_index=True)
            st.success(
                "Successfully Merged KPI, Audiences, and Market Data. Review the merged data below."
            )

    # Run market ranking and matching if data is ready
    if agg_kpi_df is not None:
        bt_run_market_ranking = st.button(
            label="**Confirm and Run Market Ranking 🏃‍➡**"
        )
        if bt_run_market_ranking:
            with st.spinner(
                text="Running ML Model to Calculate Market Scoring & Matching..."
            ):
                spend_cols = [c for c in list(df) if 'spend' in c.lower()]
                mm = MatchedMarketScoring(
                    df=df,
                    client_columns=client_columns,
                    audience_columns=audience_columns,
                    display_columns=[market_code, market_name],
                    covariate_columns=cov_columns,
                    market_column=market_code,
                    scoring_removed_columns=spend_cols
                )

            # Save model outputs to session state
            saved_outputs = {
                'mm': mm,
                'df': df,
                'kpi_df': kpi_df,
                'audience_columns': audience_columns,
                'client_columns': client_columns,
                'kpi_column': kpi_column,
                'market_level': market_level,
                'cov_columns': cov_columns,
                'market_code': market_code,
                'market_name': market_name,
                'spend_cols': spend_cols,
                'date_column': date_column,
            }
            st.session_state.update(saved_outputs)
            st.success("🚀 Successfully Ran Market Scoring & Matching 🚀")

    # Footer markdown with contact info
    st.markdown("***")
    st.markdown(
        "If you have any questions about the KPI or Audience data required, please reach out to Media Analytics "
        "team at MediaAnalytics@mediamonks.com"
    )
