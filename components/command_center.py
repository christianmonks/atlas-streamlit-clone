import pandas as pd
import streamlit as st
import os
from os.path import join
from pandas.api.types import is_numeric_dtype
from scripts.constants import *
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
            options= ['Select a Market Level'] + MARKET_LEVELS,  # Elimina None de las opciones
            #format_func=lambda x: 'Select a Market Level' if x is None else x,
            help="Choose a Market Level for Market Scoring & Market Matching"
        )

        column_market_name= market_level.split()[-1].lower().capitalize()

    # Expander for Target data selection
    # Expander for Target data selection
    with st.expander(label="**Target Audience**", expanded=False):

        if market_level == 'Select a Market Level':
            st.error("Please Return to the Previous Expander and select a Market Level", icon="🚨")
        else:
            # Read audience csv by market and country
            audience_file = f"{market_level.replace(' ', '_').lower()}_audience.csv"
            audience_path = join(cd, 'data', 'audience', audience_file)
            complete_audience_df = pd.read_csv(audience_path)

            # Create common buckets based on the existing columns in the DataFrame

            # Population aged 18 and over: sum specified columns
            complete_audience_df['P18+'] = (
                complete_audience_df[['total_18_to_34', 'total_35_to_49', 'total_50_to_64', 'total_65_and_over']].sum(axis=1)
            )

            # Male population aged 18 and over: sum specified columns
            complete_audience_df['M18+'] = (
                complete_audience_df[['male_18_to_34', 'male_35_to_49', 'male_50_to_64', 'male_65_and_over']].sum(axis=1)
            )

            # Female population aged 18 and over: sum specified columns
            complete_audience_df['F18+'] = (
                complete_audience_df[['female_18_to_34', 'female_35_to_49', 'female_50_to_64', 'female_65_and_over']].sum(axis=1)
            )

            # Population aged 18-49: sum specified columns
            complete_audience_df['P18-49'] = (
                complete_audience_df[['total_18_to_34', 'total_35_to_49']].sum(axis=1)
            )

            # Male population aged 18-49: sum specified columns
            complete_audience_df['M18-49'] = (
                complete_audience_df[['male_18_to_34', 'male_35_to_49']].sum(axis=1)
            )

            # Female population aged 18-49: sum specified columns
            complete_audience_df['F18-49'] = (
                complete_audience_df[['female_18_to_34', 'female_35_to_49']].sum(axis=1)
            )
            # Replace underscores in all column names and convert to title case
            complete_audience_df.columns = complete_audience_df.columns.str.replace('_', '-').str.title()
            complete_audience_df.columns = complete_audience_df.columns.str.replace('Female', 'F').str.title()
            complete_audience_df.columns = complete_audience_df.columns.str.replace('Male', 'M').str.title()
            complete_audience_df.columns = complete_audience_df.columns.str.replace('-Total', '').str.title()
            complete_audience_df.columns = complete_audience_df.columns.str.replace('Total', 'P').str.title()
            complete_audience_df.columns = complete_audience_df.columns.str.replace('To-', '').str.title()
            complete_audience_df.columns = complete_audience_df.columns.str.replace('-And-Over', '+').str.title()
            complete_audience_df.columns = complete_audience_df.columns.str.replace('Universe', 'P').str.title()
            complete_audience_df.columns = complete_audience_df.columns.str.replace('Under-5', '5-').str.title()

            # Add the names of the new common bucket columns to audience_columns
            audience_columns = list(complete_audience_df.columns)
            audience_columns = [col for col in audience_columns if col != MARKET_COLUMN and col != column_market_name]
            audience_columns.extend(audience_columns)  

            # Convert audience options to title case
            audience_columns = [option.title() for option in audience_columns]
            audience_columns = list(set(audience_columns))

            desired_order = [
                'P', 'P-5-', 'P-5-9', 'P-10-17', 'P-18-34', 'P18+', 'P-35-49', 'P18-49', 'P-50-64', 'P-65+',
                'M', 'M-5-', 'M-5-9', 'M-10-17', 'M-18-34', 'M18+', 'M-35-49', 'M18-49', 'M-50-64', 'M-65+',
                'F', 'F-5-', 'F-5-9', 'F-10-17', 'F-18-34', 'F18+', 'F-35-49', 'F18-49', 'F-50-64', 'F-65+'
            ]

            order_dict = {value: index for index, value in enumerate(desired_order)}
            audience_columns_sorted = sorted(audience_columns, key=lambda x: order_dict.get(x, float('inf')))


            default = "P"

            # Change from multiselect to selectbox for single selection
            audience_filter = st.selectbox(
                label="**Select Audience Column**",  # Changed to singular
                options=audience_columns_sorted,
                index=audience_columns_sorted.index(default) if default in audience_columns_sorted else 0,
                help="Select an audience demographic. Default set as Universe.",
                key="audience_selection"  # Add a key to manage state if needed
            )
            # if not audience_filter:
            #     audience_filter = default  # st.session_state.default

            # Ensure universe is not selected with other options
            # if default in audience_filter and len(audience_filter) > 1:
            #     audience_filter.remove(default)
            #     # st.session_state.default = selected_audience

            # Generate a filtered DataFrame
            audience_df = complete_audience_df[[MARKET_COLUMN] + [audience_filter]]

            # # Group and combine selected columns
            # if selected_audience:
            #     # Group by gender and age range
            #     gender_age_groups = {}

            #     for col in selected_audience:
            #         if '_' in col:  # Check if the column name contains an underscore
            #             gender, age_range = col.split('_', 1) # gender : male, female, total (f and m)
            #             if gender not in gender_age_groups:
            #                 gender_age_groups[gender] = []
            #             gender_age_groups[gender].append(age_range)

            #     combined_columns = {}
                
            #     for gender, age_ranges in gender_age_groups.items():
            #         # Custom sorting function to handle different formats
            #         def sort_key(age):
            #             if 'under' in age:
            #                 return (0, age)  # Treat 'under' as the lowest range
            #             elif 'over' in age:
            #                 return (float('inf'), age)  # Treat 'over' as the highest range
            #             else:
            #                 return (int(age.split('_')[0]), age)  # Numeric sorting for standard ranges
                    
            #         # Sort age ranges using the custom sort key
            #         age_ranges.sort(key=sort_key)
                    
            #         min_age = age_ranges[0].split('_')[0]  # First range
            #         max_age = age_ranges[-1].split('_')[-1]  # Last range
            #         combined_age_range = f"{gender} {min_age}-{max_age}" 
            #         combined_columns[combined_age_range] = audience_df[selected_audience].sum(axis=1)

            #     # Create a new DataFrame with the combined columns
            #     combined_df = pd.DataFrame(combined_columns)

            #     if "universe" not in selected_audience:
            #         if st.checkbox("Show Combined DataFrame"):
            #             st.dataframe(combined_df)

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

            date_column = next((col for col in kpi_df.columns if col == 'Date' or col == 'date'), None)

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


                if date_column:
                    # Select a date column if available
                    date_column_granularity = st.selectbox(
                        label="**Select the granularity level of the date column**",
                        options=['Daily', 'Weekly'],
                        help="Select the level of aggregation of the date column of the KPI."
                    )
                else:
                    date_column_granularity = None
            else:
                st.error("Confirm Market Level", icon="🚨")


    
     # Expander for incorporating additional data sources
    with st.expander(label="**Incorporating Additional Data Sources**"):
        if (kpi_df is None) or (market_level is None) or (MARKET_COLUMN not in list(kpi_df)):
            st.error("Please Return to the Previous Expander and Upload Audience and KPI Data", icon="🚨")
        else:
            # Load and process additional data
            additional_data = pd.read_csv(join(cd, 'data', 'census', f'{market_level.replace(" ", "_").lower()}_data.csv'))
            additional_data = additional_data.rename(columns=lambda col: rename_dict.get(col, col))

            if audience_df is not None:
                additional_columns_exclude = audience_columns
                additional_data = additional_data[[k for k in list(additional_data) if k not in additional_columns_exclude]]

            # Start with the list of dataframes
            # Filter out None or empty dataframes
            # Perform merging if there are any valid dataframes
            dfs_to_merge = [additional_data, client_df, audience_df]
            dfs_to_merge = [df for df in dfs_to_merge if df is not None]
            df = agg_kpi_df
            print(df.columns)
            for additional_df in dfs_to_merge:
                print(additional_df.columns)

            for additional_df in dfs_to_merge:
                df = df.merge(additional_df, on=MARKET_COLUMN, how='inner')
            
            # Drop columns with high null percentage
            null_percentage = (df.isnull().sum() / len(df)) * 100
            columns_to_drop = null_percentage[null_percentage > 10].index
            df = df.drop(columns=columns_to_drop)

            

            cov_columns = [c for c in additional_data if c not in [MARKET_COLUMN, column_market_name, TIER, kpi_column, 'Percent Rank'] ]
            cov_columns = {c: c.title().replace("_", " ") for c in cov_columns}

            df = df.rename(columns=cov_columns)
            cov_columns = [v for k, v in cov_columns.items()]
            cov_columns = list(set(cov_columns))
            # Identify high correlation covariates if KPI is numeric
            if is_numeric_dtype(kpi_df[kpi_column]):
                corr = df[cov_columns + [kpi_column]].corr()[kpi_column].reset_index()
                corr_vars = [
                    i for i in corr[corr[kpi_column] > VARIABLE_CORRELATION_THRESHOLD]['index'].tolist() \
                    if i != kpi_column and i != 'Universe'
                ]
            else:
                # Default columns for non-numeric KPI
                default_columns = DEFAULT_COLUMNS.get(column_market_name)
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
            client_columns = [col for col in client_columns if col != MARKET_COLUMN]
            final_columns = [MARKET_COLUMN, column_market_name] + client_columns + [audience_filter] + included_cov  + [kpi_column, TIER] 
            #final_columns = list(set(final_columns))
            df = df[final_columns]

            if st.checkbox("View Merged KPI, Audiences and Market Data"):
                st.dataframe(df, hide_index=True)
            st.success(
                "Successfully Merged KPI, Audiences, and Market Data. Review the merged data below."
            )

    # Run market ranking and matching if data is ready
    # The model requires the following variables:
    # 1. df: A DataFrame containing all necessary data
    #    - Includes columns: Market, Dma/State, KPI_columns,
    #      client_columns (audiences uploaded by the user), 
    #      audience_columns (audience selected by the user), and KPI_TIER (the target variable).
    #
    # 2. audience_column: The audience selected by the user.
    #
    # 3. client_columns: A list of audiences uploaded by the user.
    #    'Market' should be included in df but not in client_columns.
    #
    # 4. cov_columns: Columns with additional information.
    #
    # 5. display_columns: Includes Market and market_name: Dma/State.
    #   
    #   The model:
    # - Takes the complete df provided.
    # - Combines the three lists: audience_columns + client_columns + cov_columns.
    # - Filters df based on this comprehensive list as input for the model.
    # - Filters df using the target variable KPI_TIER by default.
    # - Runs the model.
    # - Uses display_columns to assign the model results (Market and Dma/State).
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
                    client_columns= client_columns,
                    audience_columns= [audience_filter],
                    display_columns=[MARKET_COLUMN, column_market_name],
                    covariate_columns=cov_columns,
                    market_column=MARKET_COLUMN,
                    scoring_removed_columns=spend_cols
                )

            # Save model outputs to session state
            saved_outputs = {
                'mm': mm,
                'df': df,
                'kpi_df': kpi_df,
                'audience_column': [audience_filter],
                'client_columns': client_columns,
                'kpi_column': kpi_column,
                'market_level': column_market_name,
                'cov_columns': cov_columns,
                'market_code': MARKET_COLUMN,
                'market_name': column_market_name,
                'spend_cols': spend_cols,
                'date_column': date_column
            }
            st.session_state.update(saved_outputs)
            st.success("🚀 Successfully Ran Market Scoring & Matching 🚀")

    # Footer markdown with contact info
    st.markdown("***")
    st.markdown(
        "If you have any questions about the KPI or Audience data required, please reach out to Media Analytics "
        "team at MediaAnalytics@mediamonks.com"
    )