import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX


def do_autoARIMA(dataset, fore_periods, periodicity=0):
    model = pm.auto_arima(dataset, start_p=1, start_q=1,
                          test='adf', max_p=4, max_q=4, m=periodicity,
                          start_P=0, seasonal=True,
                          d=None, D=1, trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)  # False full grid

    print(model.summary())
    sfit = model.fit(dataset)
    yfore = sfit.predict(n_periods=fore_periods)  # forecast
    return yfore


def run_statistical(dataset, periodicity, fore_periods):
    sarima_model = SARIMAX(dataset, order=(1, 0, 1),
                           seasonal_order=(0, 1, 0, periodicity))
    sfit = sarima_model.fit()

    # Final prediction of future value using the best hyperparameters
    forewrap = sfit.get_forecast(steps=fore_periods)
    yfore = forewrap.predicted_mean

    return yfore
