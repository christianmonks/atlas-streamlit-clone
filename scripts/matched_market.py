import copy
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from scripts.constants import *


class MatchedMarketScoring:
    def __init__(
        self,
        df: pd.DataFrame,
        audience_columns: list,
        covariate_columns: list,
        display_columns=[DMA_CODE, DMA_NAME],
        market_column=DMA_CODE,
        target_variable=TIER,
        scoring_removed_columns=[],
        run_model=True,
        feature_importance=None,
    ):
        """
        Initialize the MatchedMarketScoring class.

        :param df: DataFrame containing the data.
        :param audience_columns: List of columns representing audience features.
        :param standard_columns: List of columns representing standard features.
        :param target_variable: Name of the target variable column. Default is 'KPI_TIER'.
        """
        self.df = df.copy(deep=True)
        self.target_variable = target_variable
        self.display_columns = display_columns
        self.covariate_columns = covariate_columns
        self.audience_columns = audience_columns
        self.market_column = market_column
        self.model_columns = self.covariate_columns + self.audience_columns
        self.model_columns = [i for i in self.model_columns if i in list(self.df)]

        self.scoring_removed_columns = scoring_removed_columns
        if self.scoring_removed_columns:
            for c in self.scoring_removed_columns:
                print(f"-------- Column: {c} Removed from Market Scoring Purposes --------")
        if run_model:
            print("-------- Calculating Feature Importance --------")
            self.feature_importance = self.run_model()
        else:
            print("-------- Importing Feature Importance --------")
            self.feature_importance = feature_importance
            missing = [
                key for key in self.model_columns if key not in self.feature_importance
            ]
            for i in missing:
                print(f"-------- Missing Feature Importance: {i} --------")

        self.feature_weights = [v for k, v in self.feature_importance.items()]
        self.ranking_df, self.fi = self.score_markets()
        self.similar_markets = self.matching_markets()

    def run_model(self):
        """
        Perform model execution.

        This method executes the scoring process using a RandomForestClassifier model. It performs grid search
        for hyperparameter tuning, fits the best model, and returns the feature importances.

        :return: A dictionary representing feature importances.
        """
        df = self.df.dropna(subset=self.model_columns).reset_index(drop=True)
        x = df[self.model_columns]
        y = df[self.target_variable]

        param_grid = {
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
            "max_features": ["sqrt", None],
            "n_estimators": [10, 20, 50, 100, 200],
        }

        model = RandomForestClassifier(random_state=100)
        grid_search = GridSearchCV(
            estimator=model,
            scoring="accuracy",
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        grid_search.fit(x, y)
        best_model = grid_search.best_estimator_
        feature_importance = {
            list(x.columns)[i]: v for i, v in enumerate(best_model.feature_importances_)
        }
        return feature_importance

    def score_markets(self):
        """
        Evaluate market scores.

        This method calculates scores for markets based on feature importances, applies scaling, and computes overall and
        tier rankings. It returns DataFrames containing market rankings and feature importances.

        :return: Tuple comprising a DataFrame for market rankings and a DataFrame for feature importances.
        """
        df = self.df.dropna(subset=self.model_columns).reset_index(drop=True)
        fi = pd.DataFrame(
            list(self.feature_importance.items()), columns=["Feature", WEIGHT]
        ).sort_values(by=[WEIGHT], ascending=False)
        score_df = df[self.model_columns].copy()
        score_df = score_df[[c for c in list(score_df) if c not in self.scoring_removed_columns]]
        if "Cpm Cpm" in score_df.columns:
            score_df["Cpm Cpm"] = 1 / score_df["Cpm Cpm"]

        scaler = MinMaxScaler()
        x_norm = scaler.fit_transform(score_df)
        total_feat = sum([v for c,v in self.feature_importance.items() if c in list(score_df)])
        scores = np.matmul(x_norm, [v/total_feat for c,v in self.feature_importance.items() if c in list(score_df)])

        ranking_df = pd.DataFrame(x_norm, columns=list(score_df))
        ranking_df[SCORE] = list(scores)

        display_columns = self.display_columns + [self.target_variable]
        ranking_df = pd.concat([df[display_columns], ranking_df], axis=1)
        ranking_df = ranking_df.sort_values(by=[SCORE], ascending=False).reset_index(
            drop=True
        )
        ranking_df["Overall Rank"] = ranking_df[SCORE].rank(ascending=False).astype(int)
        ranking_df["Tier Rank"] = (
            ranking_df.groupby([TIER])[SCORE].rank(ascending=False).astype(int)
        )
        return ranking_df, fi

    def matching_markets(self):
        """
        Find similar markets based on ranking DataFrame.

        :return: DataFrame containing similar markets information.
        """
        similar_markets = pd.DataFrame()
        for t in set(self.ranking_df[self.target_variable]):
            market_tier = self.ranking_df[
                self.ranking_df[self.target_variable] == t
            ].reset_index(drop=True)
            market_dist = market_tier[
                [c for c in self.model_columns if c not in self.scoring_removed_columns]
            ]

            for i, v in market_dist.iterrows():
                rank = market_dist.apply(lambda row: abs(v - row), axis=1)
                markets = pd.DataFrame(
                    {
                        "KPI Tier": [t] * len(market_dist),
                        "Test Market Identifier": [
                            market_tier[f"{self.market_column}"].iloc[i]
                        ]
                        * len(market_dist),
                        "Test Market Name": [
                            market_tier[
                                f"{self.market_column.replace('Code', 'Name')}"
                            ].iloc[i]
                        ]
                        * len(market_dist),
                        "Control Market Identifier": [
                            i for i in market_tier[f"{self.market_column}"]
                        ],
                        "Control Market Name": [
                            i
                            for i in market_tier[
                                f"{self.market_column.replace('Code', 'Name')}"
                            ]
                        ],
                        "Similarity Index": list(
                            np.matmul(rank, [
                                v for c,v in self.feature_importance.items() if c not in self.scoring_removed_columns
                            ])
                        ),
                        "Test Market Score": [market_tier[SCORE].iloc[i]]
                        * len(market_dist),
                        "Control Market Score": [i for i in market_tier[SCORE]],
                    }
                )
                similar_markets = pd.concat([similar_markets, markets], axis=0)
        similar_markets = similar_markets[
            similar_markets["Test Market Identifier"]
            != similar_markets["Control Market Identifier"]
        ]

        similar_markets["Similarity Index"] = 1 - similar_markets["Similarity Index"]
        return similar_markets

# def generate_dma_data(
#         dma_census_data: dict,
#         included_datasets=[],
# ):
#     """
#     Generate DMA output based on census data.
#
#     :return: DataFrame containing DMA output
#     """
#     dma_data = copy.deepcopy(
#         dma_census_data
#     )
#
#     dataframes = {
#         k: v for k, v in dma_data.items() if k != 'CPM DMA'
#     }
#
#     if len(included_datasets) > 0:
#         dataframes = {
#             k: v for k, v in dataframes.items() if k in included_datasets
#         }
#
#     base = dma_data.get('CPM DMA')
#     for n, df in dataframes.items():
#         print(f'Adding in DataFrame: {n}')
#         nc = n.replace("DMA", "").strip()
#         if 'Median' not in n:
#             df.columns = [f"{k}_{nc.lower().replace(' ','_')}" if k not in [
#                 DMA_CODE, DMA_NAME
#             ] else k for k in list(df)]
#         else:
#             df = df.rename(columns={'median': nc})
#         base = base.merge(
#             df, on=[DMA_CODE, DMA_NAME], how='left'
#         )
#     return base

def calculate_tier(pct_rank, num_tiers):
    interval = 1 / num_tiers
    thresholds = [interval * i for i in range(1, num_tiers)]
    tiers = [f"Tier {i}" for i in range(num_tiers, 0, -1)]

    for threshold, tier in zip(thresholds, tiers):
        if pct_rank <= threshold:
            return tier
    return tiers[-1]
