import pg_helper
import pandas as pd
# import predict_model
import numpy as np
from datetime import date
from datetime import datetime
import dateutil.relativedelta
import getopt
import sys

import warnings
warnings.filterwarnings("ignore")

# Define force month year to perform prediction
force_month_year = ""
#Random select and split data csv files
random_split = 0

# Command line arguments pass to script
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "m:r:")
except:
    print("Error")

# Processing the Command line arguments
for opt, arg in opts:
    if opt in ['-m']:
        force_month_year = arg
    elif opt in ['-r']:
        random_split = arg

# Count the number of month between two date
def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

# Convert month-year string into datetime format then call diff_month function,
# Return number of month between two date
def month_year_cal(d1, d2):
    dt = datetime.strptime(d1, '%Y-%m')
    newdt = datetime(dt.year,dt.month,dt.day)
    dt1 = datetime.strptime(d2, '%Y-%m')
    newdt1 = datetime(dt1.year,dt1.month,dt1.day)
    return diff_month(newdt,newdt1)

# Process grouped dataframe to filter out abnormity data
def process_grouping(df):
    # Filter row that the max month_year of the group is equal to last month_year of test data
    df = df.loc[(df.month_year_max == d_lastmonthyear)]
    # Calculate the month between max and min of month_year
    df['monthcount'] = df.apply(lambda x: month_year_cal(x['month_year_max'], x['month_year_min']),axis=1)
    # Filter data by number transaction is more than 80% of monthcount
    df1 = df.loc[(df.y >= df.monthcount*0.8)]
    # Filter data by 20 month period and above
    df1 = df1.loc[(df1.monthcount >= 20)]
    # Drop unused column
    df1.drop('y', axis=1, inplace=True)
    df1.drop('month_year_min', axis=1, inplace=True)
    # copy column month_year_max to month_year
    df1['month_year'] = df1['month_year_max']
    # Drop unused column
    df1.drop('month_year_max', axis=1, inplace=True)
    df1.drop('monthcount', axis=1, inplace=True)
    # Return dataframe
    return df1

# Connecting to DataWarehouse
conn = pg_helper.postgresql_connect()
# Define the column name
column_names = ["debtorcode", "category", "projno", "y", "month_year"]
# Select data group by month_year,debtorcode,category,projno
sql_string = "SELECT debtorcode,category,projno, SUM(taxamount) AS amount, month_year FROM (SELECT ivz.\"DebtorCode\" AS debtorcode, ivz.\"DocDate\" AS docdate, ivdtlz.\"TaxCurrencyTaxableAmt\" AS taxamount, it.\"ItemType\" AS category, COALESCE ('AC',NULL,'AC') AS projno, to_char(ivz.\"DocDate\", 'YYYY-MM') AS month_year FROM \"IVDTL\" AS ivdtlz INNER JOIN \"IV\" AS ivz ON ivdtlz.\"DocKey\"=ivz.\"DocKey\" INNER JOIN \"Item\" AS it ON it.\"ItemCode\" = ivdtlz.\"ItemCode\" WHERE ivdtlz.\"ItemCode\" IS NOT NULL AND ivdtlz.\"TaxCurrencyTaxableAmt\" > 0 AND \"ProjNo\" IS NULL AND ivz.\"DocDate\" < date_trunc('month', CURRENT_DATE) UNION ALL SELECT debtorcode, docdate, amount AS taxamount, category, COALESCE ('UL',NULL,'UL') AS projno, to_char(docdate, 'YYYY-MM') AS month_year FROM unilever_inv WHERE invtype='Normal' AND docdate < date_trunc('month', CURRENT_DATE)) AS na WHERE debtorcode IN (SELECT * FROM (SELECT ivz.\"DebtorCode\" AS debtorcode FROM \"IVDTL\" AS ivdtlz INNER JOIN \"IV\" AS ivz ON ivdtlz.\"DocKey\"=ivz.\"DocKey\" INNER JOIN \"Item\" AS it ON it.\"ItemCode\" = ivdtlz.\"ItemCode\" WHERE ivdtlz.\"ItemCode\" IS NOT NULL AND ivdtlz.\"TaxCurrencyTaxableAmt\" > 0 AND \"ProjNo\" IS NULL AND ivz.\"DocDate\" >= date_trunc('month', CURRENT_DATE- INTERVAL '3' month) UNION ALL  SELECT debtorcode FROM unilever_inv WHERE invtype='Normal' AND docdate >= date_trunc('month', CURRENT_DATE- INTERVAL '3' MONTH)) na GROUP BY debtorcode) GROUP BY month_year,debtorcode,category,projno"

# Connect to DataWarehouse, query the data and put into dataframe
df_full = pg_helper.postgresql_to_dataframe(conn, sql_string, column_names)
# Close Datawarehouse connection
conn.close()
# Set column y as float64
df_full["y"] = df_full["y"].astype("float64")
# Get date of today
today = date.today()
# Get date of last month
d_lastmonth = today - dateutil.relativedelta.relativedelta(months=1)
# Convert date to month year format
d_lastmonthyear = d_lastmonth.strftime("%Y-%m")
# Exit app if command line arg force_month_year value is bigger than last month year
if force_month_year > d_lastmonthyear:
    print("Error month year: %s is the max month year" % (d_lastmonthyear))
    exit()

# Set the d_lastmonthyear var to force_month_year if user plan to predict different month
if force_month_year != "":
    d_lastmonthyear = force_month_year

# Filter data to all data below d_lastmonthyear
df_fmonth_year = df_full.loc[(df_full.month_year <= d_lastmonthyear)]

# Create new column to store min and max month_year
df_fmonth_year['month_year_min'] = df_fmonth_year['month_year']
df_fmonth_year['month_year_max'] = df_fmonth_year['month_year']

# Group debtorcode and count their month period, show max month_year
df_group = df_fmonth_year.groupby(['debtorcode','projno']).agg({'y':'count', 'month_year_min':'min', 'month_year_max':'max'}).reset_index()
df_groupdebtor = process_grouping(df_group)
df_groupdebtor.insert(1, "category",'All', True)

# Group category and count their month period, show max month_year
df_group = df_fmonth_year.groupby(['category','projno']).agg({'y':'count', 'month_year_min':'min', 'month_year_max':'max'}).reset_index()
df_groupcategory = process_grouping(df_group)
# df_groupcategory.insert(0, "projno",'All', True)
df_groupcategory.insert(0, "debtorcode",'All', True)

# Group debtorcode and category and count their month period, show max month_year
df_group = df_fmonth_year.groupby(['debtorcode','category','projno']).agg({'y':'count', 'month_year_min':'min', 'month_year_max':'max'}).reset_index()
df_groupdebtorcategory = process_grouping(df_group)

# Create a list of default data to be add into prediction items
data = []
data.insert(0, {'debtorcode': 'All', 'category': 'All', 'projno': 'AC', 'month_year': d_lastmonthyear})
data.insert(0, {'debtorcode': 'All', 'category': 'All', 'projno': 'UL', 'month_year': d_lastmonthyear})
data.insert(0, {'debtorcode': 'All', 'category': 'All', 'projno': 'All', 'month_year': d_lastmonthyear})

# Concatenate all group and default data into final dataframe
df_groupcountnew = pd.concat([pd.DataFrame(data), df_groupdebtor], ignore_index=True)
df_groupcountnew = pd.concat([df_groupcountnew, df_groupcategory], ignore_index=True)
df_groupcountnew = pd.concat([df_groupcountnew, df_groupdebtorcategory], ignore_index=True)

# Depending on Command line arg, default is normal split
if random_split == '1':
    # Creating a dataframe with 50% values of original dataframe
    # Randomly select the data
    data1 = df_groupcountnew.sample(frac = 0.5)
    # Creating dataframe with rest of the 50% values
    data2 = df_groupcountnew.drop(data1.index)
else:
    # Split 50% sequencely from top to bottom
    split50 = int(len(df_groupcountnew) * 0.5)
    data1 = df_groupcountnew[:split50]
    data2 = df_groupcountnew[split50:]

# Export dataframe to CSV files
data1.to_csv('data1.csv')
data2.to_csv('data2.csv')