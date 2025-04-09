"""
Economics MCP for domain-specific expertise in economic analysis.
This module provides the EconomicsMCP class for economic domain expertise.
"""

import logging
import time
import json
import os
from typing import Dict, List, Any, Optional
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from src.base_mcp import BaseMCP

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EconomicsMCP(BaseMCP):
    """
    Economics MCP for domain-specific expertise in economic analysis.
    
    This MCP provides capabilities for:
    1. Economic data retrieval and analysis
    2. Economic indicator tracking and forecasting
    3. Economic policy impact assessment
    4. Market trend analysis
    5. Economic scenario modeling
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize the EconomicsMCP.
        
        Args:
            api_keys: Dictionary of API keys for economic data sources
        """
        super().__init__(
            name="economics",
            description="Domain-specific expertise in economic analysis",
            version="1.0.0"
        )
        
        # Set API keys
        self.api_keys = api_keys or {}
        
        # Set default API keys from environment variables
        if "FRED_API_KEY" not in self.api_keys:
            self.api_keys["FRED_API_KEY"] = os.environ.get("FRED_API_KEY")
        
        if "WORLD_BANK_API_KEY" not in self.api_keys:
            self.api_keys["WORLD_BANK_API_KEY"] = os.environ.get("WORLD_BANK_API_KEY")
        
        if "IMF_API_KEY" not in self.api_keys:
            self.api_keys["IMF_API_KEY"] = os.environ.get("IMF_API_KEY")
        
        # Initialize data cache
        self.data_cache = {}
        
        # Operation handlers
        self.operation_handlers = {
            "get_economic_indicators": self._get_economic_indicators,
            "analyze_economic_trends": self._analyze_economic_trends,
            "forecast_economic_indicators": self._forecast_economic_indicators,
            "assess_policy_impact": self._assess_policy_impact,
            "generate_economic_scenarios": self._generate_economic_scenarios,
            "analyze_market_trends": self._analyze_market_trends,
            "get_economic_data": self._get_economic_data,
            "visualize_economic_data": self._visualize_economic_data
        }
        
        logger.info("Initialized EconomicsMCP")
    
    def process(self, input_data: Dict) -> Dict:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Processing results
        """
        logger.info("Processing input in EconomicsMCP")
        
        # Validate input
        if not isinstance(input_data, dict):
            error_msg = "Input must be a dictionary"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get operation type
        operation = input_data.get("operation")
        if not operation:
            error_msg = "No operation specified"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Check if operation is supported
        if operation not in self.operation_handlers:
            error_msg = f"Unsupported operation: {operation}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Handle operation
        try:
            result = self.operation_handlers[operation](input_data)
            return result
        except Exception as e:
            error_msg = f"Error processing operation {operation}: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _get_fred_data(self, series_id: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get economic data from FRED (Federal Reserve Economic Data).
        
        Args:
            series_id: FRED series ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with economic data
        """
        # Check if API key is available
        api_key = self.api_keys.get("FRED_API_KEY")
        if not api_key:
            logger.warning("No FRED API key available, using mock data")
            return self._get_mock_economic_data(series_id, start_date, end_date)
        
        # Check cache
        cache_key = f"fred_{series_id}_{start_date}_{end_date}"
        if cache_key in self.data_cache:
            logger.info(f"Using cached data for {cache_key}")
            return self.data_cache[cache_key]
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if not start_date:
            start_date_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365*5)  # 5 years
            start_date = start_date_dt.strftime("%Y-%m-%d")
        
        # Prepare API request
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date
        }
        
        # Call API
        try:
            response = requests.get(url, params=params)
            
            # Check for errors
            if response.status_code != 200:
                logger.error(f"FRED API error: {response.status_code} - {response.text}")
                return self._get_mock_economic_data(series_id, start_date, end_date)
            
            # Parse response
            data = response.json()
            
            # Convert to DataFrame
            observations = data.get("observations", [])
            df = pd.DataFrame(observations)
            
            # Process DataFrame
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            
            # Set index
            df = df.set_index("date")
            
            # Select relevant columns
            df = df[["value"]]
            
            # Rename column
            df = df.rename(columns={"value": series_id})
            
            # Cache data
            self.data_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting FRED data: {str(e)}")
            return self._get_mock_economic_data(series_id, start_date, end_date)
    
    def _get_world_bank_data(self, indicator: str, country: str = "US", start_year: int = None, end_year: int = None) -> pd.DataFrame:
        """
        Get economic data from World Bank.
        
        Args:
            indicator: World Bank indicator code
            country: Country code
            start_year: Start year
            end_year: End year
            
        Returns:
            DataFrame with economic data
        """
        # Check cache
        cache_key = f"wb_{indicator}_{country}_{start_year}_{end_year}"
        if cache_key in self.data_cache:
            logger.info(f"Using cached data for {cache_key}")
            return self.data_cache[cache_key]
        
        # Set default years if not provided
        if not end_year:
            end_year = datetime.now().year - 1  # Previous year
        
        if not start_year:
            start_year = end_year - 20  # 20 years
        
        # Prepare API request
        url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
        params = {
            "date": f"{start_year}:{end_year}",
            "format": "json",
            "per_page": 100
        }
        
        # Call API
        try:
            response = requests.get(url, params=params)
            
            # Check for errors
            if response.status_code != 200:
                logger.error(f"World Bank API error: {response.status_code} - {response.text}")
                return self._get_mock_economic_data(indicator, f"{start_year}-01-01", f"{end_year}-12-31")
            
            # Parse response
            data = response.json()
            
            # Check if data is available
            if len(data) < 2 or not data[1]:
                logger.warning(f"No World Bank data available for {indicator} in {country}")
                return self._get_mock_economic_data(indicator, f"{start_year}-01-01", f"{end_year}-12-31")
            
            # Convert to DataFrame
            records = []
            for item in data[1]:
                records.append({
                    "date": item["date"],
                    "value": item["value"]
                })
            
            df = pd.DataFrame(records)
            
            # Process DataFrame
            df["date"] = pd.to_datetime(df["date"], format="%Y")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            
            # Set index
            df = df.set_index("date")
            
            # Sort index
            df = df.sort_index()
            
            # Rename column
            df = df.rename(columns={"value": indicator})
            
            # Cache data
            self.data_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting World Bank data: {str(e)}")
            return self._get_mock_economic_data(indicator, f"{start_year}-01-01", f"{end_year}-12-31")
    
    def _get_imf_data(self, dataset: str, indicator: str, country: str = "US", start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get economic data from IMF (International Monetary Fund).
        
        Args:
            dataset: IMF dataset code
            indicator: IMF indicator code
            country: Country code
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with economic data
        """
        # Check cache
        cache_key = f"imf_{dataset}_{indicator}_{country}_{start_date}_{end_date}"
        if cache_key in self.data_cache:
            logger.info(f"Using cached data for {cache_key}")
            return self.data_cache[cache_key]
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if not start_date:
            start_date_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365*5)  # 5 years
            start_date = start_date_dt.strftime("%Y-%m-%d")
        
        # For now, use mock data as IMF API requires registration
        df = self._get_mock_economic_data(indicator, start_date, end_date)
        
        # Cache data
        self.data_cache[cache_key] = df
        
        return df
    
    def _get_mock_economic_data(self, indicator: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Generate mock economic data for testing without API access.
        
        Args:
            indicator: Indicator name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with mock economic data
        """
        logger.info(f"Generating mock economic data for {indicator}")
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if not start_date:
            start_date_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365*5)  # 5 years
            start_date = start_date_dt.strftime("%Y-%m-%d")
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq="M")
        
        # Create DataFrame
        df = pd.DataFrame(index=date_range)
        
        # Generate mock data based on indicator
        if "gdp" in indicator.lower():
            # GDP-like data (growing with quarterly seasonality)
            base_value = 20000
            growth_rate = 0.005  # 0.5% monthly growth
            seasonal_factor = 0.02  # 2% seasonal variation
            
            values = []
            for i, date in enumerate(date_range):
                trend = base_value * (1 + growth_rate) ** i
                seasonal = 1 + seasonal_factor * np.sin(2 * np.pi * i / 12)
                noise = np.random.normal(1, 0.005)
                value = trend * seasonal * noise
                values.append(value)
            
            df[indicator] = values
            
        elif "inflation" in indicator.lower() or "cpi" in indicator.lower():
            # Inflation-like data (fluctuating around a mean)
            base_value = 2.0  # 2% inflation
            amplitude = 1.0  # 1% variation
            
            values = []
            for i, date in enumerate(date_range):
                trend = base_value + 0.0002 * i  # Slight upward trend
                cyclical = amplitude * np.sin(2 * np.pi * i / 48)  # 4-year cycle
                noise = np.random.normal(0, 0.2)
                value = trend + cyclical + noise
                values.append(max(0, value))  # Ensure non-negative
            
            df[indicator] = values
            
        elif "unemployment" in indicator.lower():
            # Unemployment-like data (countercyclical)
            base_value = 5.0  # 5% unemployment
            amplitude = 2.0  # 2% variation
            
            values = []
            for i, date in enumerate(date_range):
                trend = base_value - 0.0001 * i  # Slight downward trend
                cyclical = amplitude * np.sin(2 * np.pi * i / 60 + np.pi)  # 5-year cycle, inverted
                noise = np.random.normal(0, 0.1)
                value = trend + cyclical + noise
                values.append(max(2, value))  # Ensure at least 2%
            
            df[indicator] = values
            
        elif "interest" in indicator.lower() or "rate" in indicator.lower():
            # Interest rate-like data (step changes with persistence)
            base_value = 2.0  # 2% interest rate
            
            values = []
            current_value = base_value
            for i, date in enumerate(date_range):
                # Occasional step changes
                if i % 6 == 0 and np.random.random() < 0.3:
                    step = np.random.choice([-0.5, -0.25, 0.25, 0.5])
                    current_value += step
                
                # Ensure non-negative
                current_value = max(0, current_value)
                
                # Add small noise
                noise = np.random.normal(0, 0.05)
                value = current_value + noise
                values.append(max(0, value))
            
            df[indicator] = values
            
        else:
            # Generic economic indicator
            base_value = 100
            growth_rate = 0.002  # 0.2% monthly growth
            
            values = []
            for i, date in enumerate(date_range):
                trend = base_value * (1 + growth_rate) ** i
                cyclical = 1 + 0.05 * np.sin(2 * np.pi * i / 60)  # 5-year cycle
                noise = np.random.normal(1, 0.01)
                value = trend * cyclical * noise
                values.append(value)
            
            df[indicator] = values
        
        return df
    
    def _get_economic_indicators(self, input_data: Dict) -> Dict:
        """
        Get economic indicators for a country or region.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Economic indicators
        """
        logger.info("Getting economic indicators")
        
        # Get parameters
        country = input_data.get("country", "US")
        indicators = input_data.get("indicators", ["gdp", "inflation", "unemployment", "interest_rate"])
        start_date = input_data.get("start_date")
        end_date = input_data.get("end_date")
        
        # Map indicator names to data source IDs
        indicator_mapping = {
            "gdp": {"source": "fred", "id": "GDP"},
            "gdp_growth": {"source": "fred", "id": "A191RL1Q225SBEA"},
            "inflation": {"source": "fred", "id": "CPIAUCSL"},
            "core_inflation": {"source": "fred", "id": "CPILFESL"},
            "unemployment": {"source": "fred", "id": "UNRATE"},
            "interest_rate": {"source": "fred", "id": "FEDFUNDS"},
            "10y_treasury": {"source": "fred", "id": "DGS10"},
            "industrial_production": {"source": "fred", "id": "INDPRO"},
            "retail_sales": {"source": "fred", "id": "RSAFS"},
            "housing_starts": {"source": "fred", "id": "HOUST"},
            "consumer_sentiment": {"source": "fred", "id": "UMCSENT"},
            "pce": {"source": "fred", "id": "PCE"},
            "gini": {"source": "worldbank", "id": "SI.POV.GINI"},
            "gdp_per_capita": {"source": "worldbank", "id": "NY.GDP.PCAP.CD"}
        }
        
        # Get data for each indicator
        results = {}
        for indicator in indicators:
            if indicator in indicator_mapping:
                source = indicator_mapping[indicator]["source"]
                source_id = indicator_mapping[indicator]["id"]
                
                if source == "fred":
                    df = self._get_fred_data(source_id, start_date, end_date)
                elif source == "worldbank":
                    # Convert dates to years for World Bank
                    start_year = int(start_date.split("-")[0]) if start_date else None
                    end_year = int(end_date.split("-")[0]) if end_date else None
                    df = self._get_world_bank_data(source_id, country, start_year, end_year)
                elif source == "imf":
                    dataset = indicator_mapping[indicator].get("dataset", "IFS")
                    df = self._get_imf_data(dataset, source_id, country, start_date, end_date)
                else:
                    logger.warning(f"Unknown data source: {source}")
                    continue
                
                # Store data
                results[indicator] = {
                    "data": df.to_dict(),
                    "metadata": {
                        "source": source,
                        "source_id": source_id,
                        "country": country,
                        "start_date": start_date,
                        "end_date": end_date
                    }
                }
            else:
                logger.warning(f"Unknown indicator: {indicator}")
        
        # Compile results
        return {
            "indicators": results,
            "country": country,
            "timestamp": time.time()
        }
    
    def _analyze_economic_trends(self, input_data: Dict) -> Dict:
        """
        Analyze economic trends based on indicator data.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Economic trend analysis
        """
        logger.info("Analyzing economic trends")
        
        # Get parameters
        country = input_data.get("country", "US")
        indicators = input_data.get("indicators", ["gdp", "inflation", "unemployment", "interest_rate"])
        start_date = input_data.get("start_date")
        end_date = input_data.get("end_date")
        
        # Get indicator data
        indicator_data = self._get_economic_indicators({
            "country": country,
            "indicators": indicators,
            "start_date": start_date,
            "end_date": end_date
        })
        
        # Analyze trends for each indicator
        trend_analysis = {}
        for indicator, data in indicator_data.get("indicators", {}).items():
            # Convert data to DataFrame
            df_dict = data.get("data", {})
            if not df_dict:
                continue
            
            # Reconstruct DataFrame
            df = pd.DataFrame(df_dict)
            
            # Calculate trends
            try:
                # Get latest value
                latest_value = df.iloc[-1].iloc[0]
                
                # Calculate change over period
                first_value = df.iloc[0].iloc[0]
                total_change = latest_value - first_value
                percent_change = (total_change / first_value) * 100 if first_value != 0 else float('inf')
                
                # Calculate recent trend (last 6 periods)
                recent_data = df.iloc[-6:].iloc[:, 0].values
                recent_trend = "stable"
                if len(recent_data) >= 3:
                    # Calculate slope of recent data
                    x = np.arange(len(recent_data))
                    slope, _, _, _, _ = np.polyfit(x, recent_data, 1, full=True)
                    
                    # Determine trend direction
                    if slope > 0.01 * np.mean(recent_data):
                        recent_trend = "increasing"
                    elif slope < -0.01 * np.mean(recent_data):
                        recent_trend = "decreasing"
                
                # Calculate volatility
                volatility = df.iloc[:, 0].std() / df.iloc[:, 0].mean() if df.iloc[:, 0].mean() != 0 else 0
                
                # Calculate seasonality
                seasonality = "none"
                if len(df) >= 24:  # Need at least 2 years of data
                    # Calculate autocorrelation
                    autocorr = pd.Series(df.iloc[:, 0].values).autocorr(lag=12)
                    if autocorr > 0.5:
                        seasonality = "strong"
                    elif autocorr > 0.3:
                        seasonality = "moderate"
                    elif autocorr > 0.1:
                        seasonality = "weak"
                
                # Store analysis
                trend_analysis[indicator] = {
                    "latest_value": float(latest_value),
                    "total_change": float(total_change),
                    "percent_change": float(percent_change),
                    "recent_trend": recent_trend,
                    "volatility": float(volatility),
                    "seasonality": seasonality
                }
                
            except Exception as e:
                logger.error(f"Error analyzing trends for {indicator}: {str(e)}")
                trend_analysis[indicator] = {"error": str(e)}
        
        # Analyze correlations between indicators
        correlations = {}
        try:
            # Combine all indicators into a single DataFrame
            combined_df = None
            for indicator, data in indicator_data.get("indicators", {}).items():
                # Convert data to DataFrame
                df_dict = data.get("data", {})
                if not df_dict:
                    continue
                
                # Reconstruct DataFrame
                df = pd.DataFrame(df_dict)
                
                # Add to combined DataFrame
                if combined_df is None:
                    combined_df = df
                else:
                    combined_df = pd.concat([combined_df, df], axis=1)
            
            # Calculate correlations
            if combined_df is not None and len(combined_df.columns) > 1:
                corr_matrix = combined_df.corr()
                
                # Convert to dictionary
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        corr_value = corr_matrix.iloc[i, j]
                        
                        correlations[f"{col1}_{col2}"] = float(corr_value)
        
        except Exception as e:
            logger.error(f"Error calculating correlations: {str(e)}")
            correlations = {"error": str(e)}
        
        # Compile results
        return {
            "trend_analysis": trend_analysis,
            "correlations": correlations,
            "country": country,
            "timestamp": time.time()
        }
    
    def _forecast_economic_indicators(self, input_data: Dict) -> Dict:
        """
        Forecast economic indicators using time series models.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Economic indicator forecasts
        """
        logger.info("Forecasting economic indicators")
        
        # Get parameters
        country = input_data.get("country", "US")
        indicators = input_data.get("indicators", ["gdp", "inflation", "unemployment", "interest_rate"])
        start_date = input_data.get("start_date")
        end_date = input_data.get("end_date")
        forecast_periods = input_data.get("forecast_periods", 12)
        
        # Get indicator data
        indicator_data = self._get_economic_indicators({
            "country": country,
            "indicators": indicators,
            "start_date": start_date,
            "end_date": end_date
        })
        
        # Forecast each indicator
        forecasts = {}
        for indicator, data in indicator_data.get("indicators", {}).items():
            # Convert data to DataFrame
            df_dict = data.get("data", {})
            if not df_dict:
                continue
            
            # Reconstruct DataFrame
            df = pd.DataFrame(df_dict)
            
            # Forecast using simple exponential smoothing
            try:
                # Get historical data
                historical_data = df.iloc[:, 0].values
                
                # Simple exponential smoothing
                alpha = 0.3  # Smoothing parameter
                
                # Initialize forecast with last observed value
                last_value = historical_data[-1]
                forecast_values = [last_value]
                
                # Generate forecasts
                for i in range(1, forecast_periods):
                    # Forecast next value
                    next_value = alpha * historical_data[-1] + (1 - alpha) * forecast_values[-1]
                    forecast_values.append(next_value)
                
                # Generate forecast dates
                last_date = pd.to_datetime(df.index[-1])
                forecast_dates = []
                
                # Determine frequency
                if len(df) >= 2:
                    date_diff = (pd.to_datetime(df.index[-1]) - pd.to_datetime(df.index[-2])).days
                    
                    if date_diff <= 1:
                        freq = "D"  # Daily
                    elif date_diff <= 7:
                        freq = "W"  # Weekly
                    elif date_diff <= 31:
                        freq = "M"  # Monthly
                    elif date_diff <= 92:
                        freq = "Q"  # Quarterly
                    else:
                        freq = "Y"  # Yearly
                else:
                    freq = "M"  # Default to monthly
                
                # Generate forecast dates
                for i in range(1, forecast_periods + 1):
                    if freq == "D":
                        next_date = last_date + timedelta(days=i)
                    elif freq == "W":
                        next_date = last_date + timedelta(weeks=i)
                    elif freq == "M":
                        next_date = last_date + pd.DateOffset(months=i)
                    elif freq == "Q":
                        next_date = last_date + pd.DateOffset(months=i*3)
                    else:
                        next_date = last_date + pd.DateOffset(years=i)
                    
                    forecast_dates.append(next_date.strftime("%Y-%m-%d"))
                
                # Store forecast
                forecasts[indicator] = {
                    "forecast_values": [float(val) for val in forecast_values],
                    "forecast_dates": forecast_dates,
                    "confidence_intervals": {
                        "lower_80": [float(val * 0.9) for val in forecast_values],
                        "upper_80": [float(val * 1.1) for val in forecast_values],
                        "lower_95": [float(val * 0.8) for val in forecast_values],
                        "upper_95": [float(val * 1.2) for val in forecast_values]
                    },
                    "method": "exponential_smoothing",
                    "parameters": {"alpha": alpha}
                }
                
            except Exception as e:
                logger.error(f"Error forecasting {indicator}: {str(e)}")
                forecasts[indicator] = {"error": str(e)}
        
        # Compile results
        return {
            "forecasts": forecasts,
            "country": country,
            "forecast_periods": forecast_periods,
            "timestamp": time.time()
        }
    
    def _assess_policy_impact(self, input_data: Dict) -> Dict:
        """
        Assess the impact of economic policies.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Policy impact assessment
        """
        logger.info("Assessing policy impact")
        
        # Get parameters
        country = input_data.get("country", "US")
        policy_type = input_data.get("policy_type", "monetary")
        policy_change = input_data.get("policy_change", "rate_increase")
        magnitude = input_data.get("magnitude", 0.25)
        
        # Define policy impact models
        policy_impacts = {
            "monetary": {
                "rate_increase": {
                    "gdp": -0.2 * magnitude,
                    "inflation": -0.3 * magnitude,
                    "unemployment": 0.1 * magnitude,
                    "stock_market": -0.5 * magnitude,
                    "housing_market": -0.4 * magnitude,
                    "consumer_spending": -0.3 * magnitude,
                    "business_investment": -0.4 * magnitude,
                    "lag_periods": {
                        "gdp": 2,
                        "inflation": 4,
                        "unemployment": 3,
                        "stock_market": 1,
                        "housing_market": 2,
                        "consumer_spending": 2,
                        "business_investment": 2
                    }
                },
                "rate_decrease": {
                    "gdp": 0.2 * magnitude,
                    "inflation": 0.3 * magnitude,
                    "unemployment": -0.1 * magnitude,
                    "stock_market": 0.5 * magnitude,
                    "housing_market": 0.4 * magnitude,
                    "consumer_spending": 0.3 * magnitude,
                    "business_investment": 0.4 * magnitude,
                    "lag_periods": {
                        "gdp": 2,
                        "inflation": 4,
                        "unemployment": 3,
                        "stock_market": 1,
                        "housing_market": 2,
                        "consumer_spending": 2,
                        "business_investment": 2
                    }
                },
                "qe": {
                    "gdp": 0.3 * magnitude,
                    "inflation": 0.2 * magnitude,
                    "unemployment": -0.2 * magnitude,
                    "stock_market": 0.7 * magnitude,
                    "housing_market": 0.5 * magnitude,
                    "consumer_spending": 0.3 * magnitude,
                    "business_investment": 0.4 * magnitude,
                    "lag_periods": {
                        "gdp": 3,
                        "inflation": 5,
                        "unemployment": 4,
                        "stock_market": 1,
                        "housing_market": 3,
                        "consumer_spending": 2,
                        "business_investment": 3
                    }
                },
                "qt": {
                    "gdp": -0.3 * magnitude,
                    "inflation": -0.2 * magnitude,
                    "unemployment": 0.2 * magnitude,
                    "stock_market": -0.7 * magnitude,
                    "housing_market": -0.5 * magnitude,
                    "consumer_spending": -0.3 * magnitude,
                    "business_investment": -0.4 * magnitude,
                    "lag_periods": {
                        "gdp": 3,
                        "inflation": 5,
                        "unemployment": 4,
                        "stock_market": 1,
                        "housing_market": 3,
                        "consumer_spending": 2,
                        "business_investment": 3
                    }
                }
            },
            "fiscal": {
                "tax_cut": {
                    "gdp": 0.4 * magnitude,
                    "inflation": 0.2 * magnitude,
                    "unemployment": -0.3 * magnitude,
                    "stock_market": 0.3 * magnitude,
                    "housing_market": 0.2 * magnitude,
                    "consumer_spending": 0.5 * magnitude,
                    "business_investment": 0.4 * magnitude,
                    "lag_periods": {
                        "gdp": 2,
                        "inflation": 3,
                        "unemployment": 3,
                        "stock_market": 1,
                        "housing_market": 2,
                        "consumer_spending": 1,
                        "business_investment": 2
                    }
                },
                "tax_increase": {
                    "gdp": -0.4 * magnitude,
                    "inflation": -0.1 * magnitude,
                    "unemployment": 0.3 * magnitude,
                    "stock_market": -0.3 * magnitude,
                    "housing_market": -0.2 * magnitude,
                    "consumer_spending": -0.5 * magnitude,
                    "business_investment": -0.4 * magnitude,
                    "lag_periods": {
                        "gdp": 2,
                        "inflation": 3,
                        "unemployment": 3,
                        "stock_market": 1,
                        "housing_market": 2,
                        "consumer_spending": 1,
                        "business_investment": 2
                    }
                },
                "spending_increase": {
                    "gdp": 0.5 * magnitude,
                    "inflation": 0.3 * magnitude,
                    "unemployment": -0.4 * magnitude,
                    "stock_market": 0.2 * magnitude,
                    "housing_market": 0.1 * magnitude,
                    "consumer_spending": 0.3 * magnitude,
                    "business_investment": 0.2 * magnitude,
                    "lag_periods": {
                        "gdp": 1,
                        "inflation": 2,
                        "unemployment": 2,
                        "stock_market": 1,
                        "housing_market": 3,
                        "consumer_spending": 1,
                        "business_investment": 2
                    }
                },
                "spending_cut": {
                    "gdp": -0.5 * magnitude,
                    "inflation": -0.2 * magnitude,
                    "unemployment": 0.4 * magnitude,
                    "stock_market": -0.2 * magnitude,
                    "housing_market": -0.1 * magnitude,
                    "consumer_spending": -0.3 * magnitude,
                    "business_investment": -0.2 * magnitude,
                    "lag_periods": {
                        "gdp": 1,
                        "inflation": 2,
                        "unemployment": 2,
                        "stock_market": 1,
                        "housing_market": 3,
                        "consumer_spending": 1,
                        "business_investment": 2
                    }
                }
            }
        }
        
        # Check if policy type and change are valid
        if policy_type not in policy_impacts:
            error_msg = f"Invalid policy type: {policy_type}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        if policy_change not in policy_impacts[policy_type]:
            error_msg = f"Invalid policy change: {policy_change}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get policy impact model
        impact_model = policy_impacts[policy_type][policy_change]
        
        # Compile results
        return {
            "policy_impact": impact_model,
            "country": country,
            "policy_type": policy_type,
            "policy_change": policy_change,
            "magnitude": magnitude,
            "timestamp": time.time()
        }
    
    def _generate_economic_scenarios(self, input_data: Dict) -> Dict:
        """
        Generate economic scenarios based on different assumptions.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Economic scenarios
        """
        logger.info("Generating economic scenarios")
        
        # Get parameters
        country = input_data.get("country", "US")
        indicators = input_data.get("indicators", ["gdp", "inflation", "unemployment", "interest_rate"])
        num_scenarios = input_data.get("num_scenarios", 3)
        forecast_periods = input_data.get("forecast_periods", 12)
        
        # Get indicator forecasts
        forecasts = self._forecast_economic_indicators({
            "country": country,
            "indicators": indicators,
            "forecast_periods": forecast_periods
        })
        
        # Define scenario parameters
        scenario_params = {
            "baseline": {
                "name": "Baseline",
                "description": "Continuation of current trends",
                "probability": 0.6,
                "adjustments": {}
            },
            "optimistic": {
                "name": "Optimistic",
                "description": "Stronger growth, lower inflation and unemployment",
                "probability": 0.2,
                "adjustments": {
                    "gdp": 0.5,
                    "inflation": -0.3,
                    "unemployment": -0.5,
                    "interest_rate": 0.2
                }
            },
            "pessimistic": {
                "name": "Pessimistic",
                "description": "Weaker growth, higher inflation and unemployment",
                "probability": 0.2,
                "adjustments": {
                    "gdp": -0.5,
                    "inflation": 0.5,
                    "unemployment": 0.7,
                    "interest_rate": -0.3
                }
            },
            "stagflation": {
                "name": "Stagflation",
                "description": "Low growth with high inflation",
                "probability": 0.1,
                "adjustments": {
                    "gdp": -0.7,
                    "inflation": 1.0,
                    "unemployment": 0.5,
                    "interest_rate": 0.5
                }
            },
            "deflation": {
                "name": "Deflation",
                "description": "Declining prices and economic contraction",
                "probability": 0.1,
                "adjustments": {
                    "gdp": -1.0,
                    "inflation": -1.5,
                    "unemployment": 1.0,
                    "interest_rate": -0.5
                }
            }
        }
        
        # Select scenarios based on num_scenarios
        selected_scenarios = ["baseline"]
        if num_scenarios >= 3:
            selected_scenarios.extend(["optimistic", "pessimistic"])
        if num_scenarios >= 4:
            selected_scenarios.append("stagflation")
        if num_scenarios >= 5:
            selected_scenarios.append("deflation")
        
        # Generate scenarios
        scenarios = {}
        for scenario_key in selected_scenarios:
            scenario = scenario_params[scenario_key]
            
            # Create scenario
            scenarios[scenario_key] = {
                "name": scenario["name"],
                "description": scenario["description"],
                "probability": scenario["probability"],
                "indicators": {}
            }
            
            # Adjust forecasts for each indicator
            for indicator, forecast in forecasts.get("forecasts", {}).items():
                if "error" in forecast:
                    continue
                
                # Get adjustment factor
                adjustment = scenario["adjustments"].get(indicator, 0)
                
                # Adjust forecast values
                adjusted_values = []
                for i, value in enumerate(forecast["forecast_values"]):
                    # Progressive adjustment (stronger in later periods)
                    period_adjustment = adjustment * (i + 1) / len(forecast["forecast_values"])
                    
                    # Apply adjustment
                    if indicator in ["gdp", "inflation"]:
                        # Percentage point adjustment for rates
                        adjusted_value = value + period_adjustment
                    else:
                        # Percentage adjustment for levels
                        adjusted_value = value * (1 + period_adjustment)
                    
                    adjusted_values.append(adjusted_value)
                
                # Store adjusted forecast
                scenarios[scenario_key]["indicators"][indicator] = {
                    "forecast_values": adjusted_values,
                    "forecast_dates": forecast["forecast_dates"]
                }
        
        # Compile results
        return {
            "scenarios": scenarios,
            "country": country,
            "forecast_periods": forecast_periods,
            "timestamp": time.time()
        }
    
    def _analyze_market_trends(self, input_data: Dict) -> Dict:
        """
        Analyze market trends based on economic indicators.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Market trend analysis
        """
        logger.info("Analyzing market trends")
        
        # Get parameters
        country = input_data.get("country", "US")
        markets = input_data.get("markets", ["stock", "bond", "housing", "commodity"])
        
        # Define market models
        market_models = {
            "stock": {
                "indicators": ["gdp_growth", "interest_rate", "inflation", "unemployment"],
                "weights": [0.4, -0.3, -0.2, -0.1],
                "baseline": 0.08,  # 8% annual return
                "volatility": 0.15  # 15% annual volatility
            },
            "bond": {
                "indicators": ["interest_rate", "inflation", "gdp_growth"],
                "weights": [0.6, -0.3, 0.1],
                "baseline": 0.04,  # 4% annual return
                "volatility": 0.06  # 6% annual volatility
            },
            "housing": {
                "indicators": ["interest_rate", "gdp_growth", "unemployment"],
                "weights": [-0.4, 0.3, -0.3],
                "baseline": 0.05,  # 5% annual return
                "volatility": 0.08  # 8% annual volatility
            },
            "commodity": {
                "indicators": ["inflation", "gdp_growth", "interest_rate"],
                "weights": [0.5, 0.3, -0.2],
                "baseline": 0.06,  # 6% annual return
                "volatility": 0.20  # 20% annual volatility
            }
        }
        
        # Get economic indicators
        all_indicators = set()
        for market in markets:
            if market in market_models:
                all_indicators.update(market_models[market]["indicators"])
        
        indicator_data = self._get_economic_indicators({
            "country": country,
            "indicators": list(all_indicators)
        })
        
        # Analyze market trends
        market_trends = {}
        for market in markets:
            if market not in market_models:
                logger.warning(f"Unknown market: {market}")
                continue
            
            model = market_models[market]
            
            # Calculate market score
            score = model["baseline"]
            
            for i, indicator in enumerate(model["indicators"]):
                if indicator in indicator_data.get("indicators", {}):
                    indicator_info = indicator_data["indicators"][indicator]
                    trend_info = self._analyze_economic_trends({
                        "country": country,
                        "indicators": [indicator]
                    })
                    
                    if indicator in trend_info.get("trend_analysis", {}):
                        # Get recent trend
                        trend = trend_info["trend_analysis"][indicator]["recent_trend"]
                        
                        # Convert trend to numeric value
                        trend_value = 0
                        if trend == "increasing":
                            trend_value = 1
                        elif trend == "decreasing":
                            trend_value = -1
                        
                        # Apply weight
                        score += trend_value * model["weights"][i] * 0.01
            
            # Determine market outlook
            outlook = "neutral"
            if score > model["baseline"] + 0.02:
                outlook = "bullish"
            elif score > model["baseline"] + 0.01:
                outlook = "moderately bullish"
            elif score < model["baseline"] - 0.02:
                outlook = "bearish"
            elif score < model["baseline"] - 0.01:
                outlook = "moderately bearish"
            
            # Calculate expected return
            expected_return = score
            
            # Calculate risk level
            risk_level = "medium"
            if model["volatility"] > 0.15:
                risk_level = "high"
            elif model["volatility"] < 0.08:
                risk_level = "low"
            
            # Store market trend
            market_trends[market] = {
                "outlook": outlook,
                "expected_return": float(expected_return),
                "risk_level": risk_level,
                "volatility": float(model["volatility"]),
                "key_drivers": model["indicators"]
            }
        
        # Compile results
        return {
            "market_trends": market_trends,
            "country": country,
            "timestamp": time.time()
        }
    
    def _get_economic_data(self, input_data: Dict) -> Dict:
        """
        Get raw economic data for specific indicators.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Economic data
        """
        logger.info("Getting economic data")
        
        # Get parameters
        country = input_data.get("country", "US")
        indicators = input_data.get("indicators", ["gdp", "inflation", "unemployment", "interest_rate"])
        start_date = input_data.get("start_date")
        end_date = input_data.get("end_date")
        
        # Get indicator data
        indicator_data = self._get_economic_indicators({
            "country": country,
            "indicators": indicators,
            "start_date": start_date,
            "end_date": end_date
        })
        
        # Convert data to CSV format
        csv_data = {}
        for indicator, data in indicator_data.get("indicators", {}).items():
            # Convert data to DataFrame
            df_dict = data.get("data", {})
            if not df_dict:
                continue
            
            # Reconstruct DataFrame
            df = pd.DataFrame(df_dict)
            
            # Convert to CSV
            csv_data[indicator] = df.to_csv()
        
        # Compile results
        return {
            "data": csv_data,
            "country": country,
            "start_date": start_date,
            "end_date": end_date,
            "timestamp": time.time()
        }
    
    def _visualize_economic_data(self, input_data: Dict) -> Dict:
        """
        Visualize economic data.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Visualization data
        """
        logger.info("Visualizing economic data")
        
        # Get parameters
        country = input_data.get("country", "US")
        indicators = input_data.get("indicators", ["gdp", "inflation", "unemployment", "interest_rate"])
        start_date = input_data.get("start_date")
        end_date = input_data.get("end_date")
        chart_type = input_data.get("chart_type", "line")
        
        # Get indicator data
        indicator_data = self._get_economic_indicators({
            "country": country,
            "indicators": indicators,
            "start_date": start_date,
            "end_date": end_date
        })
        
        # Create visualizations
        visualizations = {}
        for indicator, data in indicator_data.get("indicators", {}).items():
            # Convert data to DataFrame
            df_dict = data.get("data", {})
            if not df_dict:
                continue
            
            # Reconstruct DataFrame
            df = pd.DataFrame(df_dict)
            
            # Create visualization
            try:
                # Set up figure
                plt.figure(figsize=(10, 6))
                
                # Create chart
                if chart_type == "line":
                    plt.plot(df.index, df.iloc[:, 0], marker='o', linestyle='-')
                elif chart_type == "bar":
                    plt.bar(df.index, df.iloc[:, 0])
                elif chart_type == "area":
                    plt.fill_between(df.index, df.iloc[:, 0])
                else:
                    plt.plot(df.index, df.iloc[:, 0], marker='o', linestyle='-')
                
                # Add title and labels
                plt.title(f"{indicator.replace('_', ' ').title()} - {country}")
                plt.xlabel("Date")
                plt.ylabel("Value")
                
                # Add grid
                plt.grid(True, alpha=0.3)
                
                # Rotate x-axis labels
                plt.xticks(rotation=45)
                
                # Tight layout
                plt.tight_layout()
                
                # Save figure to bytes
                import io
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                # Convert to base64
                import base64
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                
                # Store visualization
                visualizations[indicator] = {
                    "image_data": img_str,
                    "format": "png",
                    "encoding": "base64"
                }
                
                # Close figure
                plt.close()
                
            except Exception as e:
                logger.error(f"Error creating visualization for {indicator}: {str(e)}")
                visualizations[indicator] = {"error": str(e)}
        
        # Compile results
        return {
            "visualizations": visualizations,
            "country": country,
            "chart_type": chart_type,
            "timestamp": time.time()
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of this MCP.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "operations": list(self.operation_handlers.keys()),
            "data_sources": ["FRED", "World Bank", "IMF"],
            "indicators": [
                "gdp", "gdp_growth", "inflation", "core_inflation", "unemployment",
                "interest_rate", "10y_treasury", "industrial_production", "retail_sales",
                "housing_starts", "consumer_sentiment", "pce", "gini", "gdp_per_capita"
            ]
        }
