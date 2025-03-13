from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast(
    forecast: pd.DataFrame, x_label: str = "Date", y_label: str = "Total / £"
):
    plt.plot(forecast["x"], forecast["y"], marker="o", linestyle="", label="Actuals")
    p = plt.plot(
        forecast["x"],
        forecast["y_forecast"],
        marker="o",
        label="Forecasts",
    )
    color = p[0].get_color()
    plt.fill_between(
        forecast["x"],
        forecast["y_forecast_lower"],
        forecast["y_forecast_upper"],
        alpha=0.2,
        color=color,
    )
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
