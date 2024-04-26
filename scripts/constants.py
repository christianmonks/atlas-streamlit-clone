TIER = 'KPI Tier'
PERCENT_RANK = 'Percent Rank'
FEATURE = 'Feature'
WEIGHT = 'Weight'
CPM = 'CPM'
SCORE = 'Score'
RANK = 'Rank'
DMA_CODE = 'DMA Code'
COUNTRY_CODE = 'Country Code'
COUNTRY_NAME = 'Country Name'
DMA_NAME = 'DMA Name'
DEFAULT_DMA = ['CPM DMA', 'Education Short DMA', 'Employment Status DMA', 'Internet DMA', 'Tenure Population DMA',
               'Per Capita Income DMA', 'Median Household Income DMA']
INCLUDE_DMA = ['cpm_dma.csv', 'education_short_dma.csv', 'employment_status_dma.csv', 'gdp_dma.csv', 'gini_dma.csv',
               'internet_dma.csv', 'median_household_income_dma.csv', 'median_housing_value_dma.csv',
               'median_monthly_housing_cost_dma.csv', 'per_capita_income_dma.csv', 'population_dma.csv',
               'tenure_population_dma.csv']
DMA_COLUMNS = {
    "cpm_dma.csv": ['DMA Code', 'DMA Name', 'CPM'],
    "education_short_dma.csv": ['DMA Code', 'DMA Name', 'total_bachelors_or_higher'],
    "employment_status_dma.csv": ['DMA Code', 'DMA Name', 'in_labor_force'],
    "gdp_dma.csv": ['DMA Code', 'DMA Name', '2022'],
    "gini_dma.csv": ['DMA Code', 'DMA Name', 'gini_index'],
    "internet_dma.csv": ['DMA Code', 'DMA Name', 'broadband_any_source'],
    "median_household_income_dma.csv": ['DMA Code', 'DMA Name', 'median'],
    "median_housing_value_dma.csv": ['DMA Code', 'DMA Name', 'median'],
    "median_monthly_housing_cost_dma.csv": ['DMA Code', 'DMA Name', 'median'],
    "per_capita_income_dma.csv": ['DMA Code', 'DMA Name', 'per_capita_income'],
    "population_dma.csv": ['DMA Code', 'DMA Name', 'universe'],
    "tenure_population_dma.csv": ['DMA Code', 'DMA Name', 'owner_occupied']
}
