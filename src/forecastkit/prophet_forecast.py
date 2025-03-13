import sys

from loguru import logger
import numpy as np
import pandas as pd
from prophet import Prophet


def create_past(x_values: np.array, y_values: np.array) -> pd.DataFrame:
    data = {"ds": x_values, "y": y_values}
    logger.debug(f"{data = }")
    past = pd.DataFrame(data)
    logger.debug(f"{past = }")
    return past


def create_model(scaling: str = "minmax") -> Prophet:
    logger.debug(f"{scaling = }")
    model = Prophet(scaling=scaling)
    logger.debug(f"{model = }")
    # model.add_country_holidays(country_name="UK")
    return model


def create_forecast(
    x_values: np.array,
    y_values: np.array,
    horizon: int,
    scaling: str = "minmax",
    frequency: str = "MS",
):
    # Create a model
    model = create_model(scaling=scaling)

    # Train the model
    past = create_past(x_values=x_values, y_values=y_values)
    model.fit(past)

    # Make predictions
    future = model.make_future_dataframe(periods=horizon, freq=frequency)
    forecast = model.predict(future)
    forecast = forecast.merge(past, on="ds", how="left")
    forecast = forecast[["ds", "y", "yhat", "yhat_upper", "yhat_lower"]]
    forecast = forecast.rename(
        {
            "ds": "x",
            "y": "y",
            "yhat": "y_forecast",
            "yhat_upper": "y_forecast_upper",
            "yhat_lower": "y_forecast_lower",
        },
        axis=1,
    )
    return forecast
