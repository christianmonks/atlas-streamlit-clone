import json
import streamlit as st
import pandas as pd
from pandas.api.types import is_numeric_dtype
import plotly.express as px
from scripts.constants import *
import streamlit_vertical_slider as svs
from scripts.matched_market import MatchedMarketScoring


def render_matched_markets():
    """
    Renders the matched markets UI and analysis in the Streamlit app.

    This function allows users to select tiers, exclude markets, and specify test markets for
    identifying similar matched markets. It aggregates key performance indicators (KPIs) for
    test and control markets and visualizes the matched market results.

    Functionality:
    - Users can filter markets based on tiers and specific market selections.
    - The function calculates matched market pairs based on similarity indices.
    - It generates scatter and line/bar plots for visualizing KPIs across matched markets.
    """

    # Extract required session state values
    mm1, kpi_column, market_level, date_column, market_code, df, kpi_df = (
        st.session_state[key] for key in [
            "mm1", "kpi_column", "market_level",
            "date_column", "market_code", "df", "kpi_df"
        ]
    )

    mm_df = mm1.similar_markets

    # Create columns for user inputs
    col1, col2, col3, col4 = st.columns([1, 1, 1, 0.6], gap="small")
    with col1:
        # Select tiers for matched market identification
        tier_filter = st.multiselect(
            label="**Select Tiers for Identifying Similar Matched Markets**",
            options=sorted(list(set(mm_df[TIER]))),
            default=sorted(list(set(mm_df[TIER])))[0],
            help="Choose the tiers you want to use for identifying similar matched markets.",
        )
        # Create a mask for selected tiers
        tier_mask = mm_df["KPI Tier"].isin(tier_filter)

    with col2:
        # Select markets to exclude from the test
        market_removal = st.multiselect(
            label="**Select Markets to Exclude from Test**",
            options=list(set(mm_df[tier_mask]["Test Market Name"])),
            help="Choose markets that you want to exclude from matched market pairing.",
        )
        # Create a mask for removal of selected markets
        removal_mask = (mm_df["Test Market Name"].isin(market_removal)) | \
                       (mm_df["Control Market Name"].isin(market_removal))

    with col3:
        # Select specific test markets to include
        specific_markets = st.multiselect(
            label="**Select Specific Test Markets**",
            options=list(set(mm_df[tier_mask & ~removal_mask]["Test Market Name"])),
            help="Choose specific test markets to include for matched market pairing.",
        )
        # Create a mask for specific test markets
        spec_mask = (
            mm_df["Test Market Name"].isin(specific_markets)
            if specific_markets
            else ~mm_df["Test Market Name"].isin([])
        )

    with col4:
        # Input for the number of market pairs to generate
        num_pairs = st.number_input(
            "**Number of Market Pairs**",
            min_value=1,
            max_value=10,
            value=5 if len(specific_markets) == 0 else len(specific_markets),
            help="Specify the number of test and control market pairs to generate.",
        )

    # Filter the dataframe based on selected filters
    mm_df = mm_df[tier_mask & ~removal_mask & spec_mask]
    col1, col2, col3 = st.columns([1, 5, 1], gap="medium")

    with col2:
        # If there are selected tiers, proceed with matching
        if len(tier_filter) > 0:
            counter = 0
            utilized_markets = []
            matched_df = pd.DataFrame()

            # Determine maximum number of pairs to match
            max_counter = (
                num_pairs * len(tier_filter)
                if len(specific_markets) == 0
                else len(specific_markets) * num_pairs
            )

            # Match markets based on similarity index
            while counter < max_counter:
                control_market_mask = ~mm_df["Control Market Name"].isin(utilized_markets)
                test_market_mask = ~mm_df["Test Market Name"].isin(utilized_markets)

                mm_df1 = mm_df[control_market_mask & test_market_mask].sort_values(
                    by=["Test Market Score", "Similarity Index"],
                    ascending=[False, False],
                )
                mm_df1["Rank"] = mm_df1.groupby([TIER]).cumcount() + 1

                # Concatenate the matched markets to the resulting dataframe
                matched_df = pd.concat(
                    [matched_df, mm_df1[mm_df1["Rank"] == 1]], axis=0
                )

                # Update utilized markets
                utilized_markets.extend(
                    [c for c in mm_df1[mm_df1["Rank"] == 1]["Control Market Name"]] +
                    [c for c in mm_df1[mm_df1["Rank"] == 1]["Test Market Name"]]
                )
                counter += (
                    len(tier_filter)
                    if len(specific_markets) == 0
                    else len(specific_markets)
                )

            # Display matched markets
            st.write("")
            st.write("")
            st.markdown(
                f"<h5 style='text-align: center; color: black;'>Matched Markets Based on Similarity Index</h5>",
                unsafe_allow_html=True,
            )
            st.dataframe(
                matched_df[
                    [
                        TIER,
                        "Test Market Identifier",
                        "Test Market Name",
                        "Control Market Identifier",
                        "Control Market Name",
                        "Similarity Index",
                    ]
                ].sort_values(by=TIER, ascending=True),
                hide_index=True,
                use_container_width=True,
            )

    st.write("")
    st.write("")
    # If there are selected tiers, show the analysis
    if len(tier_filter) > 0:
        with st.expander("**Matched Markets Analysis**", expanded=False):
            col1, col2, col3, col4 = st.columns([0.1, 1, 1, 0.1], gap="medium")

            with col2:
                # Create a scatter plot for matched markets
                fig_scatter = px.scatter(
                    matched_df,
                    x="Control Market Score",
                    y="Similarity Index",
                    color="Test Market Name",
                    text="Control Market Name",
                    title="Matched Market Similarity Index vs Control Market Score",
                )

                # Update layout for the scatter plot
                fig_scatter.update_layout(width=800, height=500)
                fig_scatter.update_traces(textposition="top right")
                fig_scatter.update_layout(
                    yaxis_range=[
                        matched_df["Similarity Index"].min() - 0.1,
                        matched_df["Similarity Index"].max() + 0.1,
                    ]
                )
                fig_scatter.update_layout(
                    xaxis_range=[
                        matched_df["Control Market Score"].min() - 0.2,
                        matched_df["Control Market Score"].max() + 0.2,
                    ]
                )
                st.plotly_chart(
                    fig_scatter, theme="streamlit", use_container_width=True
                )

            with col3:
                

                # Merge KPI data for comparison
                kpi_comp = kpi_df.merge(df[[market_code, TIER]], on=market_code)

                print("Before grouping:")
                print(kpi_comp.head())
                
                if is_numeric_dtype(kpi_df[kpi_column]):
                    # Aggregate KPI for test markets
                    test = (
                        kpi_comp[
                            kpi_comp[market_code].isin(matched_df["Test Market Identifier"])
                        ]
                        .groupby([date_column])[kpi_column]
                        .sum()
                        .reset_index()
                        if date_column is not None
                        else kpi_comp[
                            kpi_comp[market_code].isin(matched_df["Test Market Identifier"])
                        ]
                        .groupby(TIER)[kpi_column]
                        .sum()
                        .reset_index()
                    )

                    # Aggregate KPI for control markets
                    control = (
                        kpi_comp[
                            kpi_comp[market_code].isin(matched_df["Control Market Identifier"])
                        ]
                        .groupby([date_column])[kpi_column]
                        .sum()
                        .reset_index()
                        if date_column is not None
                        else kpi_comp[
                            kpi_comp[market_code].isin(matched_df["Control Market Identifier"])
                        ]
                        .groupby(TIER)[kpi_column]
                        .sum()
                        .reset_index()
                    )

                    # Rename columns for clarity
                    test = test.rename(columns={kpi_column: "Test Markets"})
                    control = control.rename(columns={kpi_column: "Control Markets"})

                    # Merge test and control dataframes
                    kpi_comp = (
                        test.merge(control, on=[date_column], how="left")
                        if date_column is not None
                        else test.merge(control, on=[TIER], how="left")
                    )
                    # Reshape the dataframe for plotting
                    kpi_comp = (
                        pd.melt(
                            kpi_comp,
                            id_vars=[date_column],
                            var_name="Market",
                            value_name=kpi_column,
                        )
                        if date_column is not None
                        else pd.melt(
                            kpi_comp,
                            id_vars=[TIER],
                            var_name="Market",
                            value_name=kpi_column,
                        )
                    )
                    # If date columns are present, create a line plot
                    if date_column is not None:
                        kpi_comp[date_column] = pd.to_datetime(kpi_comp[date_column])
                        kpi_comp = kpi_comp.sort_values(by=date_column, ascending=False)
                        kpi_comp = kpi_comp.rename(
                            columns={date_column: date_column.title()}
                        )
                        fig_comp = px.line(
                            kpi_comp,
                            x=date_column.title(),
                            y=kpi_column,
                            color="Market",
                            color_discrete_sequence=["blue", "red"],
                            title="Historical KPI Volume: Control vs Test Markets",
                        )
                        fig_comp.update_layout(width=800, height=500)
                        st.plotly_chart(fig_comp, theme="streamlit", use_container_width=True)
                    else:
                        fig_comp = (
                            px.bar(
                                kpi_comp,
                                x="Market",
                                y=kpi_column,
                                color="Market",
                                color_discrete_sequence=["blue", "red"],
                                title=f"Historical {kpi_column} Volume: Control vs Test Markets",
                            )
                            .update_traces(marker_line_width=0)
                            .update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
                        )
                        fig_comp.update_layout(width=800, height=500)
                        st.plotly_chart(fig_comp, theme="streamlit", use_container_width=True)
