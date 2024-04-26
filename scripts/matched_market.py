import copy
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from scripts.constants import*

class MatchedMarketScoring:
    def __init__(
            self,
            df: pd.DataFrame,
            audience_columns: list,
            covariate_columns: list,
            display_columns=[DMA_CODE, DMA_NAME],
            market_column=DMA_CODE,
            target_variable=TIER
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

        self.ranking_df, self.fi, self.model_columns, self.feature_weights = self.run_scoring()
        self.similar_markets = self.matching_markets()


    def run_scoring(self):
        """
        Run the scoring process.

        :return: Tuple containing ranking DataFrame, feature importances DataFrame, and model columns list.
        """
        model_columns = self.covariate_columns + self.audience_columns
        model_columns = [i for i in model_columns if i in list(self.df)]
        df = self.df.dropna(subset=model_columns).reset_index(drop=True)
        X = df[model_columns]
        y = df[self.target_variable]

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
            {FEATURE: feature_names, WEIGHT: feature_weights}
        ).sort_values(by=[WEIGHT], ascending=False)
        fi[f'Cumulative {WEIGHT}'] = fi[WEIGHT].cumsum()

        # calculate scores
        score_df = df[model_columns].copy()
        if 'Cpm Cpm' in score_df.columns:
            score_df['Cpm Cpm'] = 1 / score_df['Cpm Cpm']
        scaler = MinMaxScaler()
        X_norm = scaler.fit_transform(score_df)
        scores = np.matmul(X_norm, feature_weights)
        ranking_df = pd.DataFrame(X_norm, columns=model_columns)
        ranking_df[SCORE] = list(scores)

        display_columns = self.display_columns + [self.target_variable]
        ranking_df = pd.concat([df[display_columns], ranking_df], axis=1)
        ranking_df = ranking_df.sort_values(by=[SCORE], ascending=False).reset_index(drop=True)
        ranking_df['Overall Rank'] = ranking_df[SCORE].rank(ascending=False).astype(int)
        ranking_df['Tier Rank'] = ranking_df.groupby([TIER])[SCORE].rank(ascending=False).astype(int)
        return ranking_df, fi, model_columns, feature_weights

    def matching_markets(self):
        """
        Find similar markets based on ranking DataFrame.

        :return: DataFrame containing similar markets information.
        """
        similar_markets = pd.DataFrame()
        for t in set(self.ranking_df[self.target_variable]):
            market_tier = self.ranking_df[self.ranking_df[self.target_variable] == t].reset_index(drop=True)
            market_dist = market_tier[self.model_columns]

            for i, v in market_dist.iterrows():
                rank = market_dist.apply(lambda row: abs(v - row), axis=1)
                markets = pd.DataFrame(
                    {
                        'Tier': [t] * len(market_dist),
                        'Test Market Identifier': [market_tier[f'{self.market_column}'].iloc[i]] * len(market_dist),
                        'Test Market Name': [market_tier[f"{self.market_column.replace('Code', 'Name')}"].iloc[i]] * len(market_dist),
                        'Control Market Identifier':  [i for i in market_tier[f'{self.market_column}']],
                        'Control Market Name': [i for i in market_tier[f"{self.market_column.replace('Code', 'Name')}"]],
                        'Similarity Index': list(np.matmul(rank, self.feature_weights))
                    }
                )
                similar_markets = pd.concat([similar_markets, markets], axis=0)
        similar_markets = similar_markets[
            similar_markets['Test Market Identifier'] != similar_markets['Control Market Identifier']
        ]
        return similar_markets

def generate_dma_data(
        dma_census_data: dict,
        included_datasets=[],
):
    """
    Generate DMA output based on census data.

    :return: DataFrame containing DMA output
    """
    dma_data = copy.deepcopy(
        dma_census_data
    )

    dataframes = {
        k: v for k, v in dma_data.items() if k != 'CPM DMA'
    }

    if len(included_datasets) > 0:
        dataframes = {
            k: v for k, v in dataframes.items() if k in included_datasets
        }

    base = dma_data.get('CPM DMA')
    for n, df in dataframes.items():
        print(f'Adding in DataFrame: {n}')
        nc = n.replace("DMA", "").strip()
        if 'Median' not in n:
            df.columns = [f"{k}_{nc.lower().replace(' ','_')}" if k not in [
                DMA_CODE, DMA_NAME
            ] else k for k in list(df)]
        else:
            df = df.rename(columns={'median': nc})
        base = base.merge(
            df, on=[DMA_CODE, DMA_NAME], how='left'
        )
    return base

def calculate_tier(pct_rank, num_tiers):
    interval = 1 / num_tiers
    thresholds = [interval * i for i in range(1, num_tiers)]
    tiers = [f'Tier {i}' for i in range(num_tiers, 0, -1)]

    for threshold, tier in zip(thresholds, tiers):
        if pct_rank <= threshold:
            return tier
    return tiers[-1]
