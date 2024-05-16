import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from statsmodels.tsa.seasonal import seasonal_decompose
from models.machinelearning import run_RF
from models.neural import run_neural
from scipy import signal
from statsmodels.tsa.stattools import acf
from models.statistical import run_statistical, do_autoARIMA
from sklearn.metrics import mean_absolute_error


def logdiff(data):
    log_series = [np.log(x) for x in data]
    logdiff_series = [log_series[i] - log_series[i - 1] for i in range(1, len(log_series))]
    return log_series, logdiff_series


def invert_logdiff(first_item, diff_data, is_first_item_log=True):
    first_item = first_item if is_first_item_log else np.log(first_item)
    result = np.concatenate(([first_item], diff_data))
    result = np.cumsum(result)
    result = [np.exp(x) for x in result]
    return result


def preprocessing(ds):
    ds['Data'] = pd.to_datetime(ds['date']).dt.date
    ds = ds[['Data', 'meantempm']]
    ds = ds.sort_values(by='Data')

    # Calculate the moving average of the data in the 'meantempm' column using a moving window of width 7
    rolling_mean = ds['meantempm'].rolling(window=7, min_periods=1, center=True).mean()

    # Create a new DataFrame with the date corresponding to the central data and the moving average
    new_df = pd.DataFrame({'Data': ds['Data'], 'meantempm': rolling_mean})

    # Remove rows with missing values ​​(may be present due to floating windows)
    new_df.dropna(inplace=True)

    # Remove duplicate rows (to avoid duplicates due to interpolation)
    new_df.drop_duplicates(subset=['Data'], inplace=True)

    # Remove the first and last 3 days
    new_df = new_df.iloc[3:-3]

    # Set the index of the new DataFrame to the 'Data' column
    new_df.set_index('Data', inplace=True)

    print(new_df)
    print("Rows: ", new_df.shape[0])
    print("Columns: ", new_df.shape[1])

    return int(fft(new_df)[0]), new_df


def check_right_look_back(dff, cut, train_kalmann, test_kalmann):
    look_back_list1 = [12, 20, 30, 40, 42, 50, 100, 125, 150, 200]
    look_back_list2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    mse_scores = []

    for look_back in look_back_list2:
        # Esegui il modello MLP utilizzando la validazione incrociata per il valore specifico di look_back
        MLP_train_pred, MLP_fore = run_neural(dff, cut, look_back)
        MLP_fore = pd.Series(MLP_fore, index=range(len(train_kalmann), len(train_kalmann) + len(MLP_fore)))
        mse = forecast_accuracy(MLP_fore, test_kalmann)
        mse_scores.append((look_back, mse))


def fft(dataframe):
    numbers = dataframe['meantempm']
    plt.figure()
    plt.title("Jaipur Mean Temperature Data")
    plt.plot(numbers)
    plt.show()

    # the series is not stationary, in order to apply fft we need to make it stationary by detrending
    detrended_series = signal.detrend(numbers, type="linear")
    plt.figure()
    plt.title("Detrended data")
    plt.plot(detrended_series)
    plt.show()
    fft = np.fft.fft(detrended_series)
    n = len(fft)

    # for only real valued time series, fft has the conjugate complex simmetry
    # it is sufficient to consider only the first half of the values since the rest is the same
    one_sided_fft = fft[:int(n / 2) + 1]
    # sampling rate is 1: number of observation is 1 per unit of time (es., 1 obs per month)
    Fs = 1
    # nyquist theorem: is it sufficient to consider only the frequencies up to the sample rate / 2
    nyquist = Fs / 2
    # magnitude at power of 2 divided by the length
    power = np.abs(one_sided_fft) ** 2 / n
    # normalized frequency  with n/2 +1 values
    freq = np.array(range(n)) / n
    # frequencies values necessary from 0 to nyquist
    freq_to_plot = freq[:int(n / 2) + 1]
    power_max = np.amax(power)
    index = np.where(power == power_max)
    frequency_max = freq[index]
    print("Max frequency{}".format(frequency_max))
    print("Max period{}".format(1 / frequency_max))

    plt.figure()
    plt.title("Periodogram (Power Spectrum Density)")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.plot(freq_to_plot, power)
    plt.show()
    return 1 / frequency_max


def plot_seasonal_decomp(ds, p=7):
    result = seasonal_decompose(ds, model='additive', period=p)

    plt.figure(figsize=(10, 6))
    plt.subplot(4, 1, 1)
    plt.plot(result.observed, label='Observed')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(result.trend, label='Trend')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(result.seasonal, label='Seasonal')
    plt.legend()

    plt.subplot(4, 1, 4)  # it is the deviation from the sum of trends and seasons
    plt.plot(result.resid, label='Residual')
    plt.legend()

    plt.tight_layout()
    plt.show()


def k_filter(ds):
    measurements = ds.to_numpy()
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=measurements[0],
                      initial_state_covariance=1,
                      observation_covariance=10,
                      transition_covariance=10)
    st_means, state_covariances = kf.filter(measurements)
    state_std = np.sqrt(state_covariances[:, 0])
    # plot data and predictions
    plt.figure(figsize=(10, 5))
    plt.plot(measurements, label="measures")
    plt.plot(st_means, label="kalman")
    plt.legend()
    plt.title("Kalman Filter")
    plt.show()
    return st_means


def run_MLP(ds, cut, tr, tes):
    MLP_train_pred, MLP_fore = run_neural(ds, cut)
    MLP_fore = pd.Series(MLP_fore, index=range(len(tr), len(tr) + len(MLP_fore)))

    plt.plot(tr, label="train")
    plt.plot(tes, label="expected", color="darkgray")
    plt.plot(MLP_fore, label="forecast", color="green", alpha=0.5)
    plt.xlabel('time')
    plt.ylabel('degree')
    plt.title("MLP")
    plt.legend()
    plt.show()

    return MLP_fore


def run_random(ds, cut, tr, tes):
    RF_fore = run_RF(ds, 3, len(ds) - cut)
    RF_fore = pd.Series(RF_fore, index=range(len(tr), len(tr) + len(RF_fore)))

    plt.plot(tr, label="train")
    plt.plot(tes, label="expected", color="darkgray")
    plt.plot(RF_fore, label="forecast", color="red", alpha=0.5)
    plt.xlabel('time')
    plt.ylabel('degree')
    plt.title("RandomForest")
    plt.legend()
    plt.show()
    return RF_fore


def run_sarima(tr, ts, periodicity):
    SARIMA_fore = run_statistical(tr, periodicity, len(ts))
    SARIMA_fore = pd.Series(SARIMA_fore, index=range(len(tr), len(tr) + len(SARIMA_fore)))

    plt.plot(tr, label="train")
    plt.plot(ts, label="expected", color="darkgray")
    plt.plot(SARIMA_fore, label="forecast", alpha=0.5)
    plt.xlabel('time')
    plt.ylabel('degree')
    plt.title("SARIMA")
    plt.legend()
    plt.show()
    return SARIMA_fore


def run_autoarima(data, cut, tr, ts, periodicity):
    l_dataset, ld_dataset = logdiff(data)

    ld_train = ld_dataset[:cut]
    ld_test = ld_dataset[cut:]

    ld_SARIMA_fore = do_autoARIMA(ld_train, len(ld_test), periodicity)

    fore = invert_logdiff(ts.iloc[0], ld_SARIMA_fore, False)
    fore = pd.Series(fore, index=range(len(tr), len(tr) + len(fore)))

    plt.plot(tr, label="train")
    plt.plot(ts, label="expected", color="darkgray")
    plt.plot(fore, label="forecast", alpha=0.5)
    plt.legend()
    plt.show()
    return fore


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))
    me = np.mean(forecast - actual)
    mae = np.mean(np.abs(forecast - actual))
    mpe = np.mean((forecast - actual) / actual)
    rmse = np.mean((forecast - actual) ** 2) ** .5
    corr = np.corrcoef(forecast, actual)[0, 1]
    acf1 = acf(forecast - actual)[1]
    return {'mape': mape, 'me': me, 'mae': mae, 'mpe': mpe, 'rmse': rmse, 'acf1': acf1, 'corr': corr}


if __name__ == "__main__":
    # read in the csv data into a pandas data frame and set the date as the index
    original_df = pd.read_csv("dataset/JaipurWeatherData.csv")

    # Preprocessing
    stag, df_prep = preprocessing(original_df)

    # Slicing in train and test sets
    dataset = df_prep.meantempm

    cutP = int(len(dataset) * 0.7)
    plot_seasonal_decomp(dataset, stag)

    # Kalman filter
    state_means = k_filter(dataset)

    df = pd.DataFrame(state_means, columns=['meantempm'])
    df = df.meantempm
    train, test = df[:cutP], df[cutP:]

    plot_seasonal_decomp(state_means, stag)

    # STATISTICAL
    sarima_fore = run_sarima(train, test, stag)
    # fore_kalman = run_autoarima(df, cutP, train_kalman, test_kalman, stag)
    print(forecast_accuracy(sarima_fore, test))

    # check_right_look_back(df, cutP, train_kalman, test_kalman)

    # NEURAL
    mlp_fore = run_MLP(df, cutP, train, test)
    print(forecast_accuracy(mlp_fore, test))

    # MACHINE LEARNING
    random_fore = run_random(df, cutP, train, test)
    print(forecast_accuracy(random_fore, test))

    # Create a dictionary with predictions and model names
    models = {
        'SARIMA': sarima_fore,
        'MLP': mlp_fore,
        'RF': random_fore
    }

    # Calculate the MAE for each model and print the result
    mae_scores = {model_name: mean_absolute_error(test, forecast) for model_name, forecast in models.items()}
    for model_name, mae in mae_scores.items():
        print(f"{model_name} - MAE = {mae}")

    # Create the comparison DataFrame and fill with MAE values
    comparison_result = pd.DataFrame(mae_scores, index=['MAE']).transpose()

    # Find the most accurate model
    most_accurate_model = comparison_result['MAE'].idxmin()
    print("The most accurate model is {}".format(most_accurate_model))
