import numpy as np
import statsmodels.api as sm


def seasonal_cycles(values: np.array, seasonal_periods: int) -> int:
    return len(values) // seasonal_periods


def seasonal_reshape(values: np.array, seasonal_periods: int) -> np.array:
    """
    Calculates the seasonal cycles and reshapes the values into a multi-dimension array
    each of length seasonal periods.

    Args:
        values (np.array): Array of values.
        seasonal_periods (int): Number of seasonal periods.

    Returns:
        tuple[int, np.array]: Tuple containing the number of seasonal cycles and seasonal
        reshaped values.
    """
    cycles = seasonal_cycles(values, seasonal_periods)
    reshaped_values = values[-cycles * seasonal_periods :].reshape(
        cycles, seasonal_periods
    )
    return reshaped_values


def seasonal_non_zero_stats(values: np.array, seasonal_periods: int) -> tuple[np.array]:
    """
    Calculates seasonal non-zero statistics from a given array of values.

    Args:
        values (np.array): Array of values.
        seasonal_periods (int): Number of seasonal periods.

    Returns:
        tuple[np.array]: Tuple containing the seasonal non-zero probabilities and means.
    """
    cycles = seasonal_cycles(values, seasonal_periods)
    reshaped_values = seasonal_reshape(values, seasonal_periods)
    non_zero_probabilities = np.count_nonzero(reshaped_values, axis=0) / cycles
    non_zero_means = reshaped_values.mean(axis=0)
    return non_zero_probabilities, non_zero_means


def seasonal_trend(values: np.array, seasonal_periods: int) -> np.array:
    reshaped_values = seasonal_reshape(values, seasonal_periods)
    return reshaped_values.sum(axis=1)


def teunter_syntetos_babai(
    values: np.array,
    seasonal_periods: int,
    forecasting_length: int,
) -> np.array:
    """
    Generate a naive forecast based on non-zero probabilities and means.

    Args:
        non_zero_probabilities (np.array): Array of non-zero probabilities.
        non_zero_means (np.array): Array of non-zero means.
        seasonal_periods (int): Number of seasonal periods.
        forecasting_length (int): Length of the forecast.

    Returns:
        np.array: Array containing the seasonal forecast values.
    """
    non_zero_probabilities, non_zero_means = seasonal_non_zero_stats(
        values, seasonal_periods
    )
    forecast_values = non_zero_probabilities * non_zero_means
    forecast_indices = np.mod(np.arange(forecasting_length), seasonal_periods)
    return forecast_values[forecast_indices]


def holt_winters(values: np.array, seasonal_periods: int, horizon: int) -> np.array:
    """
    Generate a forecast based on non-zero probabilities and means.

    Args:
        values (np.array): Array of non-zero means.
        seasonal_periods (int): Number of seasonal periods.
        horizon (int): Length of the forecast.

    Returns:
        np.array: Array containing the seasonal forecast values.
    """
    model = sm.tsa.ExponentialSmoothing(
        values,
        seasonal_periods=seasonal_periods,
        trend="add",
        damped_trend=True,
        seasonal="add",
        use_boxcox=False,
    )
    fitted_model = model.fit(remove_bias=True, method="L-BFGS-B")
    forecast_values = fitted_model.forecast(horizon)
    return forecast_values
