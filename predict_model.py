import calendar
import numpy as np
import pandas as pd
# FBProphet package
from fbprophet import Prophet
# SARIMAX package
from pmdarima import auto_arima
# Holtwinters ExponentialSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
# XGBoost package
from numpy import asarray
from xgboost import XGBRegressor
# Validation package
from sklearn.metrics import mean_absolute_error
# PostgreSQL package
import psycopg2
import pg_helper

# Smallest possible numeric
EPSILON = 1e-10

# Prophet model
def prophet_model(df_full, forecastperiod):
    try:
        good_model = 1
        df_pr = df_full.copy()
        # Remove indexing
        df_pr = df_full.reset_index()
        # To use prophet column names should be like that
        df_pr.columns = ['ds','y']
        train_data_pr = df_pr.iloc[:len(df_pr)-forecastperiod]

        model = Prophet(seasonality_mode='multiplicative',
# Parameter modulating the flexibility of the automatic changepoint selection. Large values ​​will allow many changepoints, small values ​​will allow few changepoints. Adjust the flexibility of the "changepoint" selection. The larger the value, the more "changepoints" are selected, which makes the model fit the historical data, but it also increases the risk of overfitting. (default 0.05)
                        changepoint_prior_scale=30,
# Larger values ​​allow the model to fit larger seasonal fluctuations, smaller values ​​dampen the seasonality. Can be specified for individual seasonalities using add_seasonality. Adjust the strength of seasonal components. The larger the value, the more the seasonal fluctuations will be accommodated, and the smaller the value, the more seasonal fluctuations will be suppressed. (default 10)
                        seasonality_prior_scale=35,
# Parameter modulating the strength of the holiday components model, unless overridden in the holidays input. Adjusting the strength of the holiday model component. The larger the value, the greater the impact of the holiday on the model. The smaller the value, the smaller the impact of the holiday. (default 10)
                        holidays_prior_scale=20,
                        # Disable auto analyze the seasonality of the data
                        daily_seasonality=False,
                         # Disable auto analyze the seasonality of the data
                        weekly_seasonality=False,
                         # Disable auto analyze the seasonality of the data
                        yearly_seasonality=False)
        # Add holidays parameter into model
        model.add_country_holidays(country_name='MY')
        # Fit the model with train data
        model.fit(train_data_pr)
        # Make N number of future prediction date extending from train data
        future = model.make_future_dataframe(periods=forecastperiod, freq='MS')
        # Perform the forecasting
        data_forecast = model.predict(future)
        # Copy the forecasted value to a new series
        future_forecast = data_forecast['yhat'][-forecastperiod:].values
    except:
        future_forecast = None
        good_model = 0
    # Return the forecast value and return 1 for good_model if no exception
    # while performing forecasting
    return future_forecast, good_model

# SARIMAX model using pmdarima package
def sarimax_model(df_full, df_train, forecastperiod):
    try:
        good_model = 1
        # Fitting auto_arima model, auto_arima will perform grid search which tries various sets of p and q (also P and Q for seasonal models) parameters, selecting the model that minimizes the AIC
        stepwise_model = auto_arima(df_full, error_action='ignore',         suppress_warnings=True, seasonal=True, m=12)
        stepwise_model.fit(df_train)
        # Forecast future value using best SARIMA model return
        future_forecast = stepwise_model.predict(n_periods = forecastperiod)
    except:
        future_forecast = None
        good_model = 0
    return future_forecast, good_model

# Holt Winter Exponential Smoothing model
def holtwinter_expo(df_trainz, forecastperiod):
    model = HWES(df_trainz, seasonal_periods=6, trend='add', seasonal='mul', damped_trend=True)
    fitted = model.fit(optimized=True, use_brute=True, use_boxcox=True)
    future_forecast = fitted.forecast(steps=forecastperiod)
    return future_forecast, 1

# Persistance  Baseline prediction
def baseline_prediction_persistance(df_full, trainsize):
    # Create lagged dataset
    dataframe = pd.concat([df_full['y'].shift(1), df_full['y']], axis=1)
    dataframe.columns = ['t-1', 't+1']
    X = dataframe.values
    train_size = len(df_full) - trainsize
    test = X[train_size:]
    test_X, test_y = test[:,0], test[:,1]
    # walk-forward validation
    predictions = list()
    for x in test_X:
        yhat = x
        predictions.append(yhat)
    return predictions

# Baseline prediction model
def baseline_prediction_naive(df_train, df_test):
    dd = np.asarray(df_train['y'])
    y_hat = df_test.copy()
    y_hat['y'] = dd[len(dd)-1]
    return y_hat

# Calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred + EPSILON) / (y_true + EPSILON))) * 100

# Rescale MAPE to 0% to 100%
def rescale_mape(mape):
    yscale = 100 - mape
    if yscale < 0:
        yscale = 0
    return yscale

# Calculate MAAPE
def maape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.arctan(np.abs((y_true - y_pred + EPSILON) / (y_true + EPSILON))))

def validation_error(testvalue, forecastvalue):
    maev_error = mean_absolute_error(testvalue, forecastvalue)
    mape_error = mean_absolute_percentage_error(testvalue, forecastvalue)
    maape_error = maape(testvalue, forecastvalue)
    return  maev_error, mape_error, maape_error

def pred_trendf(lasttest, forecast):
    # 10% different
    lasttesta10 = lasttest + (lasttest*0.1)
    lasttestd10 = lasttest - (lasttest*0.1)
    pred_percentage = np.abs(((forecast-lasttest+EPSILON)/(lasttest+EPSILON)) * 100)
    trend_str = 'NC'
    if forecast >= lasttesta10:
        trend_str = 'UP'
    elif forecast <= lasttestd10:
        trend_str ='DW'
    return trend_str, pred_percentage

def pred_bt_base(rmape, rbase):
    bt_text = 'EQUAL'
    bt_percense = 0.0
    if rmape > rbase:
        bt_text = 'BETTER'
        bt_percense = rmape - rbase
    elif rmape < rbase:
        bt_text = 'WORST'
        bt_percense = rbase - rmape
    return bt_text, bt_percense

# XGBoost
# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]

# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(n_estimators=1000)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(asarray([testX]))
    return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    try:
        good_model = 1
        predictions = list()
        # split dataset
        train, test = train_test_split(data, n_test)
        # seed history with training dataset
        history = [x for x in train]
        # step over each time-step in the test set
        for i in range(len(test)):
            # split test row into input and output columns
            testX, testy = test[i, :-1], test[i, -1]
            # fit model on history and make a prediction
            yhat = xgboost_forecast(history, testX)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
            # summarize progress
            print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    except:
        good_model = 0
        predictions = None
    return predictions, good_model

# Compute last day of month
def lastday_monthyear(monthyear):
    sp_monthyear = monthyear.split ("-")
    lastday = calendar.monthrange(int(sp_monthyear[0]), int(sp_monthyear[1]))[1]
    fulldate = monthyear + "-" + str(lastday) + " 00:00:00"
    return fulldate

def demand_forecasting(df, ntest, nforecast, forecast_flag, tprediction_list):
    train_data = df[:len(df) - ntest]
    test_data = df[len(df) - ntest:]
    test_data_result = test_data.copy()

    test_meanx = test_data['y'].mean() * 5

    predict_baseline = baseline_prediction_naive(train_data, test_data)
    baseline_mae, baseline_mape, baseline_maape = validation_error(test_data, predict_baseline)
    predict_baseline = predict_baseline.clip(0,test_meanx)
    test_data_result['Baseline'] = predict_baseline

    arima_mae, arima_mape, arima_maape = None, None, None
    prophet_mae, prophet_mape, prophet_maape = None, None, None
    xgboost_mae, xgboost_mape, xgboost_maape = None, None, None
    hw_mae, hw_mape, hw_maape = None, None, None
    sarimax_testok, prophet_testok, hw_testok, xgboost_testok = 0, 0, 0, 0

    if tprediction_list.count('sarimax') > 0:
        sarimax_test, sarimax_testok = sarimax_model(df, train_data, ntest)
        if sarimax_testok == 1:
            sarimax_test = sarimax_test.clip(0,test_meanx)
            test_data_result['SARIMAX'] = sarimax_test
            test_data_result['SARIMAX'] = test_data_result['SARIMAX'].replace([np.inf, -np.inf], np.nan)
            test_data_result['SARIMAX'] = test_data_result['SARIMAX'].fillna(EPSILON)
            arima_mae, arima_mape, arima_maape = validation_error(test_data['y'], test_data_result['SARIMAX'])

    if tprediction_list.count('prophet') > 0:
        prophet_test, prophet_testok = prophet_model(df, ntest)
        if prophet_testok == 1:
            prophet_test = prophet_test.clip(0,test_meanx)
            test_data_result['Prophet'] = prophet_test
            test_data_result['Prophet'] = test_data_result['Prophet'].replace([np.inf, -np.inf], np.nan)
            test_data_result['Prophet'] = test_data_result['Prophet'].fillna(EPSILON)
            prophet_mae, prophet_mape, prophet_maape = validation_error(test_data['y'], test_data_result['Prophet'])

    if tprediction_list.count('hwexsmooth') > 0:
        hw_test, hw_testok =  holtwinter_expo(train_data, ntest)
        if hw_testok == 1:
            hw_test = hw_test.clip(0,test_meanx)
            test_data_result['HWExSmooth'] = hw_test.values
            test_data_result['HWExSmooth'] = test_data_result['HWExSmooth'].replace([np.inf, -np.inf], np.nan)
            test_data_result['HWExSmooth'] = test_data_result['HWExSmooth'].fillna(EPSILON)
            hw_mae, hw_mape, hw_maape = validation_error(test_data['y'], test_data_result['HWExSmooth'])

    if tprediction_list.count('xgboost') > 0:
        values = df.values
        # transform the time series data into supervised learning
        dataxgboost = series_to_supervised(values, n_in=6)
        # evaluate
        xgboost_test, xgboost_testok = walk_forward_validation(dataxgboost, ntest)
        if xgboost_testok == 1:
            xgboost_test = pd.Series(xgboost_test)
            xgboost_test = xgboost_test.clip(0,test_meanx)
            test_data_result['XGBoost'] = xgboost_test.values
            test_data_result['XGBoost'] = test_data_result['XGBoost'].replace([np.inf, -np.inf], np.nan)
            test_data_result['XGBoost'] = test_data_result['XGBoost'].fillna(EPSILON)
            xgboost_mae, xgboost_mape, xgboost_maape = validation_error(test_data['y'], test_data_result['XGBoost'])

    mape_errors = [arima_mape, prophet_mape, xgboost_mape, hw_mape]
    mae_errors = [arima_mae, prophet_mae, xgboost_mae, hw_mae]
    maape_errors = [arima_maape, prophet_maape, xgboost_maape, hw_maape]
    df_errors = pd.DataFrame({"Models" : ["SARIMAX", "Prophet", "XGBoost", "HWExSmooth"],"MAPE_Errors" : mape_errors, "MAE_Errors" : mae_errors, "MAAPE_Errors" : maape_errors})

    blmape_errors = [baseline_mape]
    blmae_errors = [baseline_mae]
    blmaape_errors = [baseline_maape]
    dfbaseline_errors = pd.DataFrame({"Models" : ["Baseline"],"MAPE_Errors" : blmape_errors, "MAE_Errors" : blmae_errors, "MAAPE_Errors" : blmaape_errors})

    rng = pd.date_range(df.index.max(), periods=4, freq='MS')
    rng = rng.delete(0)
    rng = pd.DataFrame({ 'ds': rng, 'val': np.random.randn(len(rng)) })
    rng = rng.set_index("ds")
    rng.drop('val', axis=1, inplace=True)

    if forecast_flag == True:
        forecast_baseline = baseline_prediction_naive(df, rng)
        forecast_baseline = forecast_baseline.clip(0,test_meanx)
        rng['Baseline'] = forecast_baseline

        if sarimax_testok == 1:
            sarimax_forecast, sarimax_forecastok  = sarimax_model(df, df, nforecast)
            sarimax_forecast = sarimax_forecast.clip(0,test_meanx)
            rng['SARIMAX'] = sarimax_forecast
            rng['SARIMAX'] = rng['SARIMAX'].replace([np.inf, -np.inf], np.nan)
            rng['SARIMAX'] = rng['SARIMAX'].fillna(EPSILON)

        if prophet_testok == 1:
            prophet_forecast, prophet_forecastok = prophet_model(df, nforecast)
            prophet_forecast = prophet_forecast.clip(0,test_meanx)
            rng['Prophet'] = prophet_forecast
            rng['Prophet'] = rng['Prophet'].replace([np.inf, -np.inf], np.nan)
            rng['Prophet'] = rng['Prophet'].fillna(EPSILON)

        if xgboost_testok == 1:
            values = df.values
            # transform the time series data into supervised learning
            dataxgboost = series_to_supervised(values, n_in=6)
            # evaluate
            xgboost_forecast, xgboost_forecastok = walk_forward_validation(dataxgboost, nforecast)
            xgboost_forecast = pd.Series(xgboost_forecast)
            xgboost_forecast = xgboost_forecast.clip(0,test_meanx)
            rng['XGBoost'] = xgboost_forecast.values
            rng['XGBoost'] = rng['XGBoost'].replace([np.inf, -np.inf], np.nan)
            rng['XGBoost'] = rng['XGBoost'].fillna(EPSILON)

        if hw_testok == 1:
            hw_forecast, hw_forecastok = holtwinter_expo(df,nforecast)
            hw_forecast = hw_forecast.clip(0,test_meanx)
            rng['HWExSmooth'] = hw_forecast.values
            rng['HWExSmooth'] = rng['HWExSmooth'].replace([np.inf, -np.inf], np.nan)
            rng['HWExSmooth'] = rng['HWExSmooth'].fillna(EPSILON)

    return df_errors, test_data_result, rng, dfbaseline_errors

def start_forecasting(df_full, tdebtorcode, tcategory, tprojno, testperiod, forecastperiod, forecast_flag, savetodb, tprediction_list):
    # Select all
    if (tdebtorcode == "All") and (tcategory == "All") and (tprojno == "All"):
        df_fullpj = df_full.copy()
    # Filter by Dataset, Unilever or AutoCount
    elif (tdebtorcode == "All") and (tcategory == "All") and (tprojno != ""):
        df_fullpj = df_full.loc[(df_full.projno == tprojno)]
    # Filter by Debtorcode
    elif (tdebtorcode != "") and (tcategory == "All"):
        df_fullpj = df_full.loc[(df_full.debtorcode == tdebtorcode)]
    # Filter by Category
    elif (tdebtorcode == "All") and (tcategory != ""):
        df_fullpj = df_full.loc[(df_full.category == tcategory)]
    # Filter by Debtorcode and Category
    elif (tdebtorcode != "") and (tcategory != ""):
        df_fullpj = df_full.loc[(df_full.debtorcode == tdebtorcode) & (df_full.category == tcategory)]

    # Group data by month_year and sum the amount
    df_groupall = df_fullpj.groupby(['month_year']).agg({'y':'sum'}).reset_index()
    # Get the max month_year
    df_lasttraindate = df_groupall['month_year'].max()
    # Create a new column ds from month_year with additional 1st day added
    # df_groupall['ds'] = df_groupall.apply(lambda x: lastday_monthyear(x['month_year']),axis=1)
    df_groupall["ds"] = df_groupall["month_year"] + "-01 00:00:00.00000"
    # Convert ds into datetime
    df_groupall['ds'] = pd.to_datetime(df_groupall['ds'])
    # Set ds as index
    df_groupall = df_groupall.set_index("ds")
    # Drop month_year
    df_groupall.drop('month_year', axis=1, inplace=True)

    # Interpolate missing month using pandas linear function
    new_date = pd.date_range(start=df_groupall.index.min(), end=df_groupall.index.max(), freq='MS')
    df = pd.DataFrame({ 'Val' : None }, index=new_date)
    df_testnew = pd.concat([df_groupall,df], axis=1)
    df_testnew.drop('Val', axis=1, inplace=True)
    if df_testnew['y'].isnull().sum() > 0:
        df_testnew = df_testnew.interpolate('linear')

    # Forecast job
    dfz_errors, dfz_test, dfz_forecast, dfzbaseline_errors = demand_forecasting(df_testnew, testperiod, forecastperiod, forecast_flag, tprediction_list)

    # Find which Prediction model has the best result
    dfz_mapemin = dfz_errors[dfz_errors.MAPE_Errors == dfz_errors.MAPE_Errors.min()]
    dfz_maapecin = dfz_errors[dfz_errors.MAAPE_Errors == dfz_errors.MAAPE_Errors.min()]

    # Test data mean() value
    v_dfz_mean = dfz_test['y'].mean()

    # Extract MAPE and MAAPE from metrics dataframe
    v_dfz_mape_model = dfz_mapemin['Models'].iloc[0]
    v_dfz_mape_min = dfz_mapemin['MAPE_Errors'].iloc[0]
    v_dfz_maape_model = dfz_maapecin['Models'].iloc[0]
    v_dfz_maape_min = dfz_maapecin['MAAPE_Errors'].iloc[0]
    # Extract MAE base on best MAPE
    dfz_maet = dfz_errors[dfz_errors.Models == v_dfz_mape_model]
    v_dfz_maev = dfz_maet['MAE_Errors'].iloc[0]

    #Extract MAPE from baseline metrics
    v_dfz_mape_baseline = dfzbaseline_errors['MAPE_Errors'].iloc[0]
    # Rescale MAPE and Baseline MAPE error to percentage
    v_dfz_rmape = rescale_mape(v_dfz_mape_min)
    v_dfz_rbaseline = rescale_mape(v_dfz_mape_baseline)
    # Compare ML model with baseline and the percentage diff between them
    v_dfz_pbb, v_dfz_pbbp = pred_bt_base(v_dfz_rmape, v_dfz_rbaseline)
    # Predict trend
    lasttest_y = dfz_test['y'].iloc[-1]
    forecast_series = dfz_forecast[v_dfz_mape_model].iloc[0]
    v_dfz_pred_trend, v_dfz_pred_trend_p = pred_trendf(lasttest_y, forecast_series)

    # Routine to save the result into DataWarehouse
    if savetodb == True:
        df_errorjson = dfz_errors.to_json()
        dfz_testjson = dfz_test.to_json()
        dfz_forecastjson = dfz_forecast.to_json()
        dfz_groupall = df_groupall.to_json()
        dfzbaseline_json = dfzbaseline_errors.to_json()
        try:
            conn = pg_helper.postgresql_connect()
            cursor = conn.cursor()

            insert_query = "INSERT INTO sales_forecast (debtorcode, category, projno, mape_model, mape_min, rmape, rbaseline, maape_model, maape_min, test_mean, metrics_df, train_df, validation_df, forecast_df, baseline_df, pred_bt_base, pred_bt_base_p, pred_trend, mae_v, last_trainmy, pred_trend_p, tmonth_amt, forecast_amt, last_update) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,  %s, %s, NOW()) ON CONFLICT (debtorcode, category, projno, last_trainmy) DO UPDATE SET mape_model=%s, mape_min=%s, rmape=%s, rbaseline=%s, maape_model=%s, maape_min=%s, test_mean=%s, metrics_df=%s, train_df=%s, validation_df=%s, forecast_df=%s, baseline_df=%s, pred_bt_base=%s, pred_bt_base_p=%s,  pred_trend=%s, mae_v=%s, pred_trend_p=%s, tmonth_amt=%s, forecast_amt=%s, last_update=NOW()"
            record_to_insert = (tdebtorcode, tcategory, tprojno, v_dfz_mape_model, v_dfz_mape_min, v_dfz_rmape, v_dfz_rbaseline, v_dfz_maape_model, v_dfz_maape_min, v_dfz_mean, df_errorjson, dfz_groupall, dfz_testjson, dfz_forecastjson, dfzbaseline_json, v_dfz_pbb, v_dfz_pbbp, v_dfz_pred_trend, v_dfz_maev, df_lasttraindate, v_dfz_pred_trend_p, lasttest_y, forecast_series, v_dfz_mape_model, v_dfz_mape_min, v_dfz_rmape, v_dfz_rbaseline, v_dfz_maape_model, v_dfz_maape_min, v_dfz_mean, df_errorjson, dfz_groupall, dfz_testjson, dfz_forecastjson, dfzbaseline_json, v_dfz_pbb, v_dfz_pbbp, v_dfz_pred_trend, v_dfz_maev, v_dfz_pred_trend_p, lasttest_y, forecast_series)

            cursor.execute(insert_query, record_to_insert)

            conn.commit()
            count = cursor.rowcount
            print (count, "Record inserted successfully into mobile table")
        except (Exception, psycopg2.Error) as error :
            if(conn):
                print("Failed to insert record into mobile table", error)
        finally:
            #closing database connection.
            if(conn):
                cursor.close()
                conn.close()
                print("PostgreSQL connection is closed")
        return 1
    else:
        return dfz_errors, dfz_test, dfz_forecast