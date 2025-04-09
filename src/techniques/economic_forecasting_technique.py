"""
Economic Forecasting Technique using Simple Moving Average (SMA).
"""

from typing import List, Dict, Any
import logging
import numpy as np
from .analytical_technique import AnalyticalTechnique

logger = logging.getLogger(__name__)

class EconomicForecastingTechnique(AnalyticalTechnique):
    """
    A technique for forecasting economic data using the Simple Moving Average (SMA) method.
    """

    def __init__(self, method: str = "SMA", window: int = 0):
        """
        Initializes the EconomicForecastingTechnique.
        """
        self.method = method
        
        super().__init__(
            name=f"Economic Forecasting ({method})",
            description=f"Forecasts economic time series data using {method}.",
            parameters={
                "time_series_data": {"type": "List[Dict]", "description": "Time series data with 'date' and 'value' keys."},
                "forecast_periods": {"type": "int", "description": "Number of periods to forecast."},                
                "window": {"type": "int", "description": "Window size for the moving average."},
                "method": {"type": "str", "description": "Forecasting method ('SMA' or 'Exponential Smoothing').", "default": "SMA"}
            },
        )
        logger.info("Initialized EconomicForecastingTechnique.")

    def process(self, **kwargs) -> Dict[str, Any]:
        """
        Processes the input data and returns an economic forecast.

        Args:
            **kwargs: Keyword arguments containing 'time_series_data', 'forecast_periods', and 'window'.

        Returns:
            A dictionary containing the forecast and related information.

        Raises:
            ValueError: If input data is invalid or forecast parameters are incorrect.
        """
        logger.info(f"Processing economic forecasting with parameters: {kwargs}")
        
        try:
            time_series_data: List[Dict] = kwargs.get("time_series_data")
            forecast_periods: int = kwargs.get("forecast_periods")
            window: int = kwargs.get("window",0)
            method: str = kwargs.get("method", "SMA")
            

            if not isinstance(time_series_data, list) or not all(
                isinstance(item, dict) and "value" in item and isinstance(item["value"], (int, float))
                for item in time_series_data
            ):
                raise ValueError("Invalid time series data format. Expected a list of dictionaries with 'value'.")

            if not isinstance(forecast_periods, int) or forecast_periods <= 0:
                raise ValueError("Invalid forecast periods. Must be a positive integer.")
            
            if method == "SMA":
                if not isinstance(window, int) or window <= 0 or window > len(time_series_data):
                    raise ValueError(
                        "Invalid window size. Must be a positive integer less than or equal to the number of data points."
                    )
                )

            # Extract values for easier calculation
            values = [item["value"] for item in time_series_data]

            # Calculate forecast
            if method == "SMA":
                forecast = self._calculate_sma_forecast(values, forecast_periods, window)
            elif method == "Exponential Smoothing":
                forecast = self._calculate_exponential_smoothing_forecast(values, forecast_periods)
            else:
                raise ValueError(f"Invalid forecasting method: {method}")

            logger.info("Economic forecasting completed successfully.")
            return {
                "method": method,
                "forecast": forecast,                
                "input_data_length": len(time_series_data),                
                "forecast_periods": forecast_periods,
                "window": window,
            }

        except ValueError as e:
            logger.error(f"Error during economic forecasting: {e}")
            raise

    def _calculate_sma_forecast(self, values: List[float], forecast_periods: int, window: int) -> List[float]:
        """
        Calculates the SMA forecast.

        Args:
            values: List of historical values.
            forecast_periods: Number of periods to forecast.
            window: Window size for the SMA.

        Returns:
            A list of forecasted values.
        """
        forecast = []
        for _ in range(forecast_periods):
            last_window = values[-window:]
            sma = sum(last_window) / window
            forecast.append(sma)
            values.append(sma)  # Append forecasted value for next period's calculation
        return forecast

    def _calculate_exponential_smoothing_forecast(self, values: List[float], forecast_periods: int) -> List[float]:
        """
        Calculates the Exponential Smoothing forecast.

        Args:
            values: List of historical values.
            forecast_periods: Number of periods to forecast.

        Returns:
            A list of forecasted values.
        """
        alpha = 0.3  # Smoothing parameter
        forecast = []
        last_value = values[-1]
        for _ in range(forecast_periods):
            next_value = alpha * last_value + (1 - alpha) * (values[-1] if len(values)>1 else last_value)
            forecast.append(next_value)
            last_value = next_value
        return forecast