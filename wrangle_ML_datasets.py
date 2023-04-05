data_url = 'https://raw.githubusercontent.com/LambdaSchool/DS-Unit-2-Applied-Modeling/master/data/'

# Read New York City property sales data
import pandas as pd
df = pd.read_csv(DATA_PATH+'condos/NYC_Citywide_Rolling_Calendar_Sales.csv')

# change column names: replace spaces with underscores
df.columns = df.columns.str.replace(' ', '_')
df.head()

# subset of data: Tribeca Neighborhood
mask = df['NEIGHBORHOOD'] == 'TRIBECA'
df = df[mask]
df.drop(columns='NEIGHBORHOOD', inplace=True)

# range of property sales in tribeca
tribeca_range = pd.to_datetime(df['SALE_DATE'])

# convert SALE_PRICE to integers
df['SALE_PRICE'] = df['SALE_PRICE'].str.strip('$-').str.replace(',','').str.replace('-','').astype(int)

# find most expensive house
max_sale_price = df[df.SALE_PRICE == df.SALE_PRICE.max()]

# value count: total units
df_value_counts = df['TOTAL_UNITS'].value_counts()
df = df[(df['TOTAL_UNITS'] == 1.0)]

# new max sales price
df_max1 = df['SALE_PRICE'].max()

# square feet of most expensive house
max_sale_price1 = df[df.SALE_PRICE == df.SALE_PRICE.max()]
sq_feet = max_sale_price1['GROSS_SQUARE_FEET']
