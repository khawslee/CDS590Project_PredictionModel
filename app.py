import pg_helper
import pandas as pd
import predict_model
import numpy as np
from datetime import date
import sys
import getopt

import warnings
warnings.filterwarnings("ignore")

# Define test period for model evaluation
test_period = 6
# Define forecast period for future predictions
forecast_period = 3
# Define data file containing list of items to forecast
data_file = "data1.csv"
# Define Predictive models to be used
prediction_model = "sarimax,prophet,xgboost,hwexsmooth"

# Command line arguments pass to script
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "t:f:d:m:")
except:
    print("Error")

# Processing the Command line arguments
for opt, arg in opts:
    if opt in ['-t']:
        test_period = arg
    elif opt in ['-f']:
        forecast_period = arg
    elif opt in ['-d']:
        data_file = arg
    elif opt in ['-m']:
        prediction_model = arg

# prediction_list contain the list of prediction model to be perform
prediction_list = prediction_model.split (",")
print(prediction_list)

# Connecting to DataWarehouse
conn = pg_helper.postgresql_connect()
# Define the column name for the data to be extracted from Datawarehouse
column_names = ["debtorcode", "docdate", "y", "category", "projno", "month_year"]

# Select data group by month_year,debtorcode,category,projno from Datawarehouse
sql_string = "SELECT * FROM (SELECT ivz.\"DebtorCode\" AS debtorcode, ivz.\"DocDate\" AS docdate, ivdtlz.\"TaxCurrencyTaxableAmt\" AS taxamount, it.\"ItemType\" AS category, COALESCE ('AC',NULL,'AC') AS projno, to_char(ivz.\"DocDate\", 'YYYY-MM') AS month_year FROM \"IVDTL\" AS ivdtlz INNER JOIN \"IV\" AS ivz ON ivdtlz.\"DocKey\"=ivz.\"DocKey\" INNER JOIN \"Item\" AS it ON it.\"ItemCode\" = ivdtlz.\"ItemCode\" WHERE ivdtlz.\"ItemCode\" IS NOT NULL AND ivdtlz.\"TaxCurrencyTaxableAmt\" > 0 AND \"ProjNo\" IS NULL AND ivz.\"DocDate\" < date_trunc('month', CURRENT_DATE) UNION ALL SELECT debtorcode, docdate, amount AS taxamount, category, COALESCE ('UL',NULL,'UL') AS projno, to_char(docdate, 'YYYY-MM') AS month_year FROM unilever_inv WHERE invtype='Normal' AND docdate < date_trunc('month', CURRENT_DATE)) AS na WHERE debtorcode IN (SELECT * FROM (SELECT ivz.\"DebtorCode\" AS debtorcode FROM \"IVDTL\" AS ivdtlz INNER JOIN \"IV\" AS ivz ON ivdtlz.\"DocKey\"=ivz.\"DocKey\" INNER JOIN \"Item\" AS it ON it.\"ItemCode\" = ivdtlz.\"ItemCode\" WHERE ivdtlz.\"ItemCode\" IS NOT NULL AND ivdtlz.\"TaxCurrencyTaxableAmt\" > 0 AND \"ProjNo\" IS NULL AND ivz.\"DocDate\" >= date_trunc('month', CURRENT_DATE- INTERVAL '3' MONTH) UNION ALL SELECT debtorcode FROM unilever_inv WHERE invtype='Normal' AND docdate >= date_trunc('month', CURRENT_DATE- INTERVAL '3' MONTH)) na GROUP BY debtorcode)"

# Connect to DataWarehouse, query the data and put into dataframe
df_full = pg_helper.postgresql_to_dataframe(conn, sql_string, column_names)

# Define the column name for the sales forecast data to be extracted from Datawarehouse
column_names = ["debtorcode", "category","month_year", "projno"]
sql_string = "SELECT debtorcode, category,last_trainmy,projno  FROM sales_forecast"
# Connect to DataWarehouse, query the data and put into dataframe
df_sales_forecast = pg_helper.postgresql_to_dataframe(conn, sql_string, column_names)
# Close Datawarehouse connection
conn.close()
# Set column y as float64
df_full["y"] = df_full["y"].astype("float64")

# Load list of debtorcode and category to be forecast
df_groupcount = pd.read_csv(data_file, index_col=0)
# Fill N/A with empty string
df_groupcount = df_groupcount.fillna('')
# Iterate each row of prediction items
for index, row in df_groupcount.iterrows():
    tdebtorcode = row["debtorcode"]
    tcategory = row["category"]
    tprojno = row["projno"]
    monthyear = row["month_year"]
    # To verify if the same forecast had been preform before
    dftemp = df_sales_forecast.loc[(df_sales_forecast.debtorcode == tdebtorcode) & (df_sales_forecast.category == tcategory) & (df_sales_forecast.month_year == monthyear) & (df_sales_forecast.projno == tprojno)]
    if len(dftemp) > 0:
        # Previous forecast found, skip the prediction job for this item
        print("Found: " + tdebtorcode + ":" + tcategory + ":" + monthyear + ":" + tprojno)
    else:
        # Perform prediction for the item
        print("DebtorCode: [" + tdebtorcode + "] Category: [" + tcategory + "] Projno: [" + tprojno + "]")
        # Filter to work on data with month_year less than prediction item month year
        df_fmonth_year = df_full.loc[(df_full.month_year <= monthyear)]
        # Start forecasting the prediction item
        predict_model.start_forecasting(df_fmonth_year, tdebtorcode, tcategory, tprojno, test_period, forecast_period, True, True, prediction_list)
