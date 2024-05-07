TIER = "KPI Tier"
PERCENT_RANK = "Percent Rank"
FEATURE = "Feature"
WEIGHT = "Weight"
CPM = "CPM"
SCORE = "Score"
RANK = "Rank"
DMA_CODE = "DMA Code"
COUNTRY_CODE = "Country Code"
COUNTRY_NAME = "Country Name"
DMA_NAME = "DMA Name"
DEFAULT_DMA = [
    "CPM DMA",
    "Education Short DMA",
    "Employment Status DMA",
    "Internet DMA",
    "Tenure Population DMA",
    "Per Capita Income DMA",
    "Median Household Income DMA",
]

DEFAULT_WORLD_COLS = [
    "Surface Area (Km2)",
    "Population In Thousands (2017)",
    "Population Density (Per Km2, 2017)",
    "Sex Ratio (M Per 100 F, 2017)",
    "Gdp: Gross Domestic Product (Million Current Us$)",
    "Gdp Growth Rate (Annual %, Const. 2005 Prices)",
    "Gdp Per Capita (Current Us$)",
    "Economy: Industry (% Of Gva)",
    "Economy: Services And Other Activity (% Of Gva)",
    "Employment: Agriculture (% Of Employed)",
    "Employment: Industry (% Of Employed)",
    "Employment: Services (% Of Employed)",
    "Unemployment (% Of Labour Force)",
    "Labour Force Participation (Female/Male Pop. %)",
    "Population Growth Rate (Average Annual %)",
    "Urban Population (% Of Total Population)",
    "Fertility Rate, Total (Live Births Per Woman)",
    "Life Expectancy At Birth (Females/Males, Years)",
    "Population Age Distribution (0-14 / 60+ Years, %)",
]

INCLUDE_DMA = [
    "cpm_dma.csv",
    "education_short_dma.csv",
    "employment_status_dma.csv",
    "gdp_dma.csv",
    "gini_dma.csv",
    "internet_dma.csv",
    "median_household_income_dma.csv",
    "median_housing_value_dma.csv",
    "median_monthly_housing_cost_dma.csv",
    "per_capita_income_dma.csv",
    "population_dma.csv",
    "tenure_population_dma.csv",
]
DMA_COLUMNS = {
    "cpm_dma.csv": ["DMA Code", "DMA Name", "CPM"],
    "education_short_dma.csv": ["DMA Code", "DMA Name", "total_bachelors_or_higher"],
    "employment_status_dma.csv": ["DMA Code", "DMA Name", "in_labor_force"],
    "gdp_dma.csv": ["DMA Code", "DMA Name", "2022"],
    "gini_dma.csv": ["DMA Code", "DMA Name", "gini_index"],
    "internet_dma.csv": ["DMA Code", "DMA Name", "broadband_any_source"],
    "median_household_income_dma.csv": ["DMA Code", "DMA Name", "median"],
    "median_housing_value_dma.csv": ["DMA Code", "DMA Name", "median"],
    "median_monthly_housing_cost_dma.csv": ["DMA Code", "DMA Name", "median"],
    "per_capita_income_dma.csv": ["DMA Code", "DMA Name", "per_capita_income"],
    "population_dma.csv": ["DMA Code", "DMA Name", "universe"],
    "tenure_population_dma.csv": ["DMA Code", "DMA Name", "owner_occupied"],
}
