[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

# Demand Forecasting of Fast Moving Consumer Goods Distributor

This project implements a demand forecasting system for a fast-moving consumer goods (FMCG) distributor, using time series modeling to predict sales amounts and product sales. The system leverages historical sales data from a PostgreSQL database and employs various forecasting models to generate predictions.

## Project Overview

The project consists of three main Python scripts:

-   `app.py`: This is the main application script that orchestrates the demand forecasting process. It connects to a PostgreSQL database, retrieves sales data, and then calls the `predict_model.py` script to perform the forecasting. It also handles command-line arguments for specifying the test period, forecast period, data file, and prediction models.
-   `predict_model.py`: This script contains the implementation of various time series forecasting models, including Prophet, SARIMAX, Holt-Winters Exponential Smoothing, and XGBoost. It also includes functions for model validation, error calculation, and saving results to a PostgreSQL database.
-   `prepare.py`: This script is responsible for preparing the data for forecasting. It connects to the PostgreSQL database, retrieves sales data, groups the data by various dimensions (debtor code, category, project number), and filters out abnormal data. It also splits the data into two CSV files (`data1.csv` and `data2.csv`) for use in the forecasting process.

## Key Features

-   **Data Retrieval:** Connects to a PostgreSQL database to retrieve historical sales data.
-   **Data Preparation:** Groups and filters data to prepare it for forecasting.
-   **Time Series Modeling:** Implements various time series forecasting models, including Prophet, SARIMAX, Holt-Winters Exponential Smoothing, and XGBoost.
-   **Model Validation:** Includes functions for model validation and error calculation.
-   **Data Storage:** Saves forecasting results back to the PostgreSQL database.
-   **Command-Line Interface:** Provides a command-line interface for specifying various parameters, such as test period, forecast period, data file, and prediction models.

## Usage

1.  Ensure you have a PostgreSQL database set up with the required sales data.
2.  Run `prepare.py` to prepare the data and generate `data1.csv` and `data2.csv`.
3.  Run `app.py` with the desired command-line arguments to perform the forecasting.

## Dependencies

-   psycopg2
-   pandas
-   numpy
-   fbprophet
-   pmdarima
-   statsmodels
-   xgboost
-   sklearn

This project provides a comprehensive solution for demand forecasting of fast-moving consumer goods, leveraging time series modeling and a PostgreSQL database.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

[contributors-shield]: https://img.shields.io/github/contributors/khawslee/CDS590Project_PredictionModel.svg?style=for-the-badge
[contributors-url]: https://github.com/khawslee/CDS590Project_PredictionModel/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/khawslee/CDS590Project_PredictionModel.svg?style=for-the-badge
[forks-url]: https://github.com/khawslee/CDS590Project_PredictionModel/network/members
[stars-shield]: https://img.shields.io/github/stars/khawslee/CDS590Project_PredictionModel.svg?style=for-the-badge
[stars-url]: https://github.com/khawslee/CDS590Project_PredictionModel/stargazers
[issues-shield]: https://img.shields.io/github/issues/khawslee/CDS590Project_PredictionModel.svg?style=for-the-badge
[issues-url]: https://github.com/khawslee/CDS590Project_PredictionModel/issues
[license-shield]: https://img.shields.io/github/license/khawslee/CDS590Project_PredictionModel.svg?style=for-the-badge
[license-url]: https://github.com/khawslee/CDS590Project_PredictionModel/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/khawslee