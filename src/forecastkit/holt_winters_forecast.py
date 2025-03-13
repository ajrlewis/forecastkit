from loguru import logger
import numpy as np
import pandas as pd
import statsmodels.api as sm

ExponentialSmoothing = sm.tsa.ExponentialSmoothing


def create_model(y_values: np.array, seasonal_periods: int) -> ExponentialSmoothing:
    model = ExponentialSmoothing(
        y_values,
        seasonal_periods=seasonal_periods,
        trend="additive",
        damped_trend=True,
        seasonal="additive",
        use_boxcox=True,
        initial_level=0.0,
        initial_trend=0.0,
        initial_seasonal=np.zeros(seasonal_periods),
        initialization_method="known",
    )
    return model


def create_forecast(
    x_values: np.array, y_values: np.array, seasonal_periods: int, horizon: int
) -> pd.DataFrame:
    """
    Generates and returns a Holt-Winters forecast.

    Args:
        x_values (np.array): Array of non-zero means.
        y_values (np.array): Array of non-zero means.
        seasonal_periods (int): Number of seasonal periods.
        horizon (int): Length of the forecast.

    Returns:
        np.array: Array containing the seasonal forecast values.
    """
    # Create the model
    print(f"{y_values = } {horizon = }")
    model = create_model(y_values=y_values, seasonal_periods=seasonal_periods)
    # Train the model
    fitted_model = model.fit(remove_bias=True, method="L-BFGS-B")
    y_train = fitted_model.fittedfcast
    # y_train = y_train[:-1]
    y_train = y_train[1:]
    logger.debug(f"{len(x_values) = } {len(y_values) = } {len(y_train) = }")
    train = pd.DataFrame({"x": x_values, "y": y_values, "y_forecast": y_train})
    y_forecast = fitted_model.forecast(steps=horizon)

    freq = pd.infer_freq(train["x"])
    last_date = train["x"].max()
    dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)
    dates = dates[1:]

    forecast = pd.DataFrame({"x": dates, "y_forecast": y_forecast})
    forecast = pd.concat([train, forecast]).reset_index(drop=True)

    print(forecast)

    import matplotlib.pyplot as plt

    plt.plot(forecast["x"], forecast["y"], label="Actuals")
    plt.plot(forecast["x"], forecast["y_forecast"], label="Forecasts")
    plt.legend()
    plt.show()

    import sys

    sys.exit()
    # Forecast with the model
    y_forecast = fitted_model.forecast(steps=horizon)
    y_forecast_upper = y_forecast * 0.0
    y_forecast_lower = y_forecast * 0.0
    data = {
        "x": x_values,
        "y": y_values,
        "y_forecast": y_forecast,
        "y_forecast_upper": y_forecast_upper,
        "y_forecast_lower": y_forecast_lower,
    }
    logger.debug()
    forecast = pd.DataFrame(data)
    return forecast
