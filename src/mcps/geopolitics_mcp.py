"""
Geopolitics MCP for domain-specific expertise in geopolitical analysis.
This module provides the GeopoliticsMCP class for geopolitical domain expertise.
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

class GeopoliticsMCP(BaseMCP):
    """
    Geopolitics MCP for domain-specific expertise in geopolitical analysis.
    
    This MCP provides capabilities for:
    1. Geopolitical risk assessment
    2. Regional stability analysis
    3. Conflict potential evaluation
    4. International relations modeling
    5. Political trend analysis
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize the GeopoliticsMCP.
        
        Args:
            api_keys: Dictionary of API keys for geopolitical data sources
        """
        super().__init__(
            name="geopolitics",
            description="Domain-specific expertise in geopolitical analysis",
            version="1.0.0"
        )
        
        # Set API keys
        self.api_keys = api_keys or {}
        
        # Set default API keys from environment variables
        if "GDELT_API_KEY" not in self.api_keys:
            self.api_keys["GDELT_API_KEY"] = os.environ.get("GDELT_API_KEY")
        
        if "ACLED_API_KEY" not in self.api_keys:
            self.api_keys["ACLED_API_KEY"] = os.environ.get("ACLED_API_KEY")
        
        # Initialize data cache
        self.data_cache = {}
        
        # Load geopolitical knowledge base
        self.knowledge_base = self._load_knowledge_base()
        
        # Operation handlers
        self.operation_handlers = {
            "assess_geopolitical_risk": self._assess_geopolitical_risk,
            "analyze_regional_stability": self._analyze_regional_stability,
            "evaluate_conflict_potential": self._evaluate_conflict_potential,
            "model_international_relations": self._model_international_relations,
            "analyze_political_trends": self._analyze_political_trends,
            "get_event_data": self._get_event_data,
            "get_country_profile": self._get_country_profile,
            "visualize_geopolitical_data": self._visualize_geopolitical_data
        }
        
        logger.info("Initialized GeopoliticsMCP")
    
    def process(self, input_data: Dict) -> Dict:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Processing results
        """
        logger.info("Processing input in GeopoliticsMCP")
        
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
    
    def _load_knowledge_base(self) -> Dict:
        """
        Load geopolitical knowledge base.
        
        Returns:
            Knowledge base dictionary
        """
        # In a real implementation, this would load from a database or file
        # For now, we'll use a hardcoded knowledge base
        
        knowledge_base = {
            "regions": {
                "north_america": {
                    "countries": ["USA", "Canada", "Mexico"],
                    "regional_organizations": ["NAFTA", "USMCA"],
                    "key_issues": ["immigration", "trade", "drug trafficking"],
                    "stability_index": 0.9
                },
                "europe": {
                    "countries": ["Germany", "France", "UK", "Italy", "Spain"],
                    "regional_organizations": ["EU", "NATO"],
                    "key_issues": ["Brexit", "migration", "economic integration", "Russia relations"],
                    "stability_index": 0.8
                },
                "east_asia": {
                    "countries": ["China", "Japan", "South Korea", "North Korea", "Taiwan"],
                    "regional_organizations": ["ASEAN+3", "RCEP"],
                    "key_issues": ["territorial disputes", "North Korea", "US-China competition"],
                    "stability_index": 0.6
                },
                "middle_east": {
                    "countries": ["Saudi Arabia", "Iran", "Israel", "Turkey", "Egypt"],
                    "regional_organizations": ["Arab League", "GCC"],
                    "key_issues": ["Iran-Saudi rivalry", "Israel-Palestine", "terrorism", "oil politics"],
                    "stability_index": 0.4
                },
                "africa": {
                    "countries": ["Nigeria", "South Africa", "Egypt", "Ethiopia", "Kenya"],
                    "regional_organizations": ["African Union", "ECOWAS"],
                    "key_issues": ["terrorism", "civil conflicts", "development", "resource competition"],
                    "stability_index": 0.5
                }
            },
            "countries": {
                "USA": {
                    "region": "north_america",
                    "government_type": "federal_republic",
                    "stability_index": 0.85,
                    "military_power": 0.95,
                    "economic_power": 0.95,
                    "diplomatic_influence": 0.95,
                    "key_allies": ["UK", "Canada", "Australia", "Japan", "South Korea"],
                    "key_rivals": ["China", "Russia", "Iran", "North Korea"],
                    "internal_issues": ["political polarization", "inequality", "racial tensions"]
                },
                "China": {
                    "region": "east_asia",
                    "government_type": "one_party_state",
                    "stability_index": 0.75,
                    "military_power": 0.85,
                    "economic_power": 0.90,
                    "diplomatic_influence": 0.80,
                    "key_allies": ["Russia", "Pakistan", "North Korea"],
                    "key_rivals": ["USA", "India", "Japan"],
                    "internal_issues": ["ethnic tensions", "environmental issues", "economic transition"]
                },
                "Russia": {
                    "region": "eurasia",
                    "government_type": "semi_authoritarian",
                    "stability_index": 0.65,
                    "military_power": 0.80,
                    "economic_power": 0.60,
                    "diplomatic_influence": 0.75,
                    "key_allies": ["China", "Belarus", "Syria", "Iran"],
                    "key_rivals": ["USA", "NATO countries", "Ukraine"],
                    "internal_issues": ["economic stagnation", "corruption", "demographic decline"]
                },
                "Germany": {
                    "region": "europe",
                    "government_type": "parliamentary_democracy",
                    "stability_index": 0.90,
                    "military_power": 0.60,
                    "economic_power": 0.85,
                    "diplomatic_influence": 0.80,
                    "key_allies": ["France", "EU members", "USA"],
                    "key_rivals": ["Russia"],
                    "internal_issues": ["immigration integration", "far-right politics", "energy transition"]
                },
                "India": {
                    "region": "south_asia",
                    "government_type": "parliamentary_democracy",
                    "stability_index": 0.70,
                    "military_power": 0.70,
                    "economic_power": 0.75,
                    "diplomatic_influence": 0.65,
                    "key_allies": ["USA", "Russia", "Japan"],
                    "key_rivals": ["China", "Pakistan"],
                    "internal_issues": ["religious tensions", "poverty", "territorial disputes"]
                }
                # Additional countries would be included in a real implementation
            },
            "global_issues": {
                "climate_change": {
                    "impact": 0.9,
                    "timeline": "long_term",
                    "key_actors": ["USA", "China", "EU", "India"],
                    "geopolitical_implications": ["resource competition", "migration", "conflict over adaptation"]
                },
                "terrorism": {
                    "impact": 0.7,
                    "timeline": "ongoing",
                    "key_actors": ["ISIS", "Al-Qaeda", "Boko Haram"],
                    "geopolitical_implications": ["regional instability", "military interventions", "security alliances"]
                },
                "nuclear_proliferation": {
                    "impact": 0.8,
                    "timeline": "ongoing",
                    "key_actors": ["North Korea", "Iran", "USA", "Russia"],
                    "geopolitical_implications": ["security dilemmas", "arms races", "diplomatic crises"]
                },
                "cyber_warfare": {
                    "impact": 0.8,
                    "timeline": "increasing",
                    "key_actors": ["USA", "China", "Russia", "North Korea", "Iran"],
                    "geopolitical_implications": ["new security threats", "intelligence competition", "critical infrastructure vulnerabilities"]
                },
                "economic_inequality": {
                    "impact": 0.7,
                    "timeline": "increasing",
                    "key_actors": ["Global institutions", "Developed nations", "Developing nations"],
                    "geopolitical_implications": ["political instability", "migration", "populism"]
                }
            },
            "conflict_zones": {
                "ukraine": {
                    "countries_involved": ["Ukraine", "Russia"],
                    "intensity": 0.8,
                    "duration": "2014-present",
                    "type": "interstate_proxy",
                    "international_involvement": ["USA", "EU", "NATO"],
                    "resolution_prospects": 0.3
                },
                "syria": {
                    "countries_involved": ["Syria", "Russia", "Iran", "Turkey", "USA"],
                    "intensity": 0.9,
                    "duration": "2011-present",
                    "type": "civil_war_proxy",
                    "international_involvement": ["Russia", "USA", "Iran", "Turkey", "Saudi Arabia"],
                    "resolution_prospects": 0.2
                },
                "yemen": {
                    "countries_involved": ["Yemen", "Saudi Arabia", "Iran"],
                    "intensity": 0.8,
                    "duration": "2014-present",
                    "type": "civil_war_proxy",
                    "international_involvement": ["Saudi Arabia", "Iran", "USA", "UAE"],
                    "resolution_prospects": 0.3
                },
                "south_china_sea": {
                    "countries_involved": ["China", "Vietnam", "Philippines", "Malaysia", "Taiwan"],
                    "intensity": 0.5,
                    "duration": "ongoing",
                    "type": "territorial_dispute",
                    "international_involvement": ["USA", "ASEAN"],
                    "resolution_prospects": 0.4
                },
                "israel_palestine": {
                    "countries_involved": ["Israel", "Palestine"],
                    "intensity": 0.7,
                    "duration": "decades",
                    "type": "territorial_ethnic_religious",
                    "international_involvement": ["USA", "Egypt", "Jordan", "Iran"],
                    "resolution_prospects": 0.2
                }
            },
            "international_relations": {
                "usa_china": {
                    "relationship_type": "strategic_competition",
                    "cooperation_level": 0.4,
                    "conflict_level": 0.6,
                    "key_issues": ["trade", "technology", "Taiwan", "South China Sea", "human rights"],
                    "trend": "deteriorating"
                },
                "usa_russia": {
                    "relationship_type": "adversarial",
                    "cooperation_level": 0.2,
                    "conflict_level": 0.8,
                    "key_issues": ["Ukraine", "Syria", "arms control", "cyber", "NATO expansion"],
                    "trend": "stable_negative"
                },
                "china_russia": {
                    "relationship_type": "strategic_partnership",
                    "cooperation_level": 0.7,
                    "conflict_level": 0.3,
                    "key_issues": ["energy", "defense", "Central Asia", "opposition to US hegemony"],
                    "trend": "improving"
                },
                "eu_russia": {
                    "relationship_type": "strained",
                    "cooperation_level": 0.3,
                    "conflict_level": 0.7,
                    "key_issues": ["Ukraine", "energy dependency", "sanctions", "cyber attacks"],
                    "trend": "deteriorating"
                },
                "india_china": {
                    "relationship_type": "competitive_coexistence",
                    "cooperation_level": 0.4,
                    "conflict_level": 0.6,
                    "key_issues": ["border disputes", "regional influence", "Pakistan", "trade"],
                    "trend": "fluctuating"
                }
            }
        }
        
        return knowledge_base
    
    def _get_gdelt_data(self, countries: List[str], event_types: List[str] = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get event data from GDELT (Global Database of Events, Language, and Tone).
        
        Args:
            countries: List of country codes
            event_types: List of event type codes
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with event data
        """
        # Check cache
        cache_key = f"gdelt_{'_'.join(countries)}_{start_date}_{end_date}"
        if cache_key in self.data_cache:
            logger.info(f"Using cached data for {cache_key}")
            return self.data_cache[cache_key]
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if not start_date:
            start_date_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)  # 30 days
            start_date = start_date_dt.strftime("%Y-%m-%d")
        
        # For now, use mock data as GDELT API can be complex
        df = self._get_mock_event_data(countries, event_types, start_date, end_date)
        
        # Cache data
        self.data_cache[cache_key] = df
        
        return df
    
    def _get_acled_data(self, countries: List[str], event_types: List[str] = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get conflict data from ACLED (Armed Conflict Location & Event Data Project).
        
        Args:
            countries: List of country codes
            event_types: List of event type codes
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with conflict data
        """
        # Check if API key is available
        api_key = self.api_keys.get("ACLED_API_KEY")
        if not api_key:
            logger.warning("No ACLED API key available, using mock data")
            return self._get_mock_event_data(countries, event_types, start_date, end_date, conflict_focused=True)
        
        # Check cache
        cache_key = f"acled_{'_'.join(countries)}_{start_date}_{end_date}"
        if cache_key in self.data_cache:
            logger.info(f"Using cached data for {cache_key}")
            return self.data_cache[cache_key]
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if not start_date:
            start_date_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)  # 30 days
            start_date = start_date_dt.strftime("%Y-%m-%d")
        
        # Prepare API request
        url = "https://api.acleddata.com/acled/read"
        
        params = {
            "key": api_key,
            "email": "user@example.com",  # Would be replaced with actual email
            "country": "|".join(countries),
            "event_date": f"{start_date}|{end_date}",
            "event_type": "|".join(event_types) if event_types else "",
            "export_format": "json"
        }
        
        # Call API
        try:
            response = requests.get(url, params=params)
            
            # Check for errors
            if response.status_code != 200:
                logger.error(f"ACLED API error: {response.status_code} - {response.text}")
                return self._get_mock_event_data(countries, event_types, start_date, end_date, conflict_focused=True)
            
            # Parse response
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data.get("data", []))
            
            # Process DataFrame
            if not df.empty:
                df["event_date"] = pd.to_datetime(df["event_date"])
                
                # Set index
                df = df.set_index("event_date")
                
                # Sort index
                df = df.sort_index()
            
            # Cache data
            self.data_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting ACLED data: {str(e)}")
            return self._get_mock_event_data(countries, event_types, start_date, end_date, conflict_focused=True)
    
    def _get_mock_event_data(self, countries: List[str], event_types: List[str] = None, start_date: str = None, end_date: str = None, conflict_focused: bool = False) -> pd.DataFrame:
        """
        Generate mock event data for testing without API access.
        
        Args:
            countries: List of country codes
            event_types: List of event type codes
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            conflict_focused: Whether to focus on conflict events
            
        Returns:
            DataFrame with mock event data
        """
        logger.info(f"Generating mock event data for {', '.join(countries)}")
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if not start_date:
            start_date_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)  # 30 days
            start_date = start_date_dt.strftime("%Y-%m-%d")
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        
        # Define event types
        if conflict_focused:
            all_event_types = [
                "Violence against civilians",
                "Armed clash",
                "Remote explosive/landmine/IED",
                "Protest",
                "Riot",
                "Strategic development",
                "Government regains territory"
            ]
        else:
            all_event_types = [
                "Diplomatic statement",
                "Diplomatic meeting",
                "Sign agreement",
                "Military movement",
                "Economic sanction",
                "Provide aid",
                "Protest",
                "Armed conflict",
                "Peace negotiation"
            ]
        
        # Filter event types if provided
        if event_types:
            filtered_event_types = [et for et in all_event_types if any(requested_et.lower() in et.lower() for requested_et in event_types)]
            if filtered_event_types:
                all_event_types = filtered_event_types
        
        # Generate mock data
        data = []
        
        for country in countries:
            # Get country info from knowledge base if available
            country_info = self.knowledge_base.get("countries", {}).get(country, {})
            stability_index = country_info.get("stability_index", 0.7)
            
            # Adjust event frequency based on stability
            event_frequency = max(0.1, 1.0 - stability_index)
            
            for date in date_range:
                # Generate random number of events for this day
                num_events = np.random.poisson(event_frequency * 3)
                
                for _ in range(num_events):
                    # Select random event type with weighting based on conflict_focused and stability
                    if conflict_focused:
                        # Higher instability means more violent events
                        weights = [
                            (1.0 - stability_index) * 0.3,  # Violence against civilians
                            (1.0 - stability_index) * 0.3,  # Armed clash
                            (1.0 - stability_index) * 0.2,  # Remote explosive
                            stability_index * 0.3,          # Protest
                            (1.0 - stability_index) * 0.1,  # Riot
                            0.1,                            # Strategic development
                            stability_index * 0.1           # Government regains territory
                        ]
                    else:
                        # Higher stability means more diplomatic events
                        weights = [
                            stability_index * 0.2,          # Diplomatic statement
                            stability_index * 0.2,          # Diplomatic meeting
                            stability_index * 0.1,          # Sign agreement
                            (1.0 - stability_index) * 0.2,  # Military movement
                            (1.0 - stability_index) * 0.1,  # Economic sanction
                            stability_index * 0.1,          # Provide aid
                            (1.0 - stability_index) * 0.2,  # Protest
                            (1.0 - stability_index) * 0.3,  # Armed conflict
                            stability_index * 0.1           # Peace negotiation
                        ]
                    
                    # Normalize weights
                    weights = [w / sum(weights) for w in weights]
                    
                    event_type = np.random.choice(all_event_types, p=weights[:len(all_event_types)])
                    
                    # Generate fatalities for conflict events
                    fatalities = 0
                    if event_type in ["Armed clash", "Violence against civilians", "Remote explosive/landmine/IED", "Armed conflict"]:
                        fatalities = np.random.poisson(2)
                    
                    # Generate event intensity
                    if "clash" in event_type.lower() or "violence" in event_type.lower() or "conflict" in event_type.lower():
                        intensity = np.random.uniform(0.6, 1.0)
                    elif "protest" in event_type.lower() or "riot" in event_type.lower():
                        intensity = np.random.uniform(0.3, 0.7)
                    else:
                        intensity = np.random.uniform(0.1, 0.5)
                    
                    # Create event record
                    event = {
                        "event_date": date,
                        "country": country,
                        "event_type": event_type,
                        "actor1": f"{country} Government" if np.random.random() < 0.6 else f"{country} Opposition",
                        "actor2": f"{country} Civilians" if np.random.random() < 0.5 else f"{country} Militants",
                        "location": f"{country} Capital" if np.random.random() < 0.3 else f"{country} Region {np.random.randint(1, 5)}",
                        "fatalities": fatalities,
                        "intensity": intensity
                    }
                    
                    data.append(event)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Set index
        if not df.empty:
            df = df.set_index("event_date")
            df = df.sort_index()
        
        return df
    
    def _assess_geopolitical_risk(self, input_data: Dict) -> Dict:
        """
        Assess geopolitical risk for countries or regions.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Geopolitical risk assessment
        """
        logger.info("Assessing geopolitical risk")
        
        # Get parameters
        countries = input_data.get("countries", [])
        regions = input_data.get("regions", [])
        time_horizon = input_data.get("time_horizon", "medium")  # short, medium, long
        
        # Validate input
        if not countries and not regions:
            error_msg = "No countries or regions specified"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Expand regions to countries
        all_countries = list(countries)
        for region in regions:
            region_info = self.knowledge_base.get("regions", {}).get(region, {})
            region_countries = region_info.get("countries", [])
            all_countries.extend(region_countries)
        
        # Remove duplicates
        all_countries = list(set(all_countries))
        
        # Get event data for risk assessment
        start_date = None
        end_date = None
        
        if time_horizon == "short":
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        elif time_horizon == "medium":
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        else:  # long
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        event_data = self._get_gdelt_data(all_countries, start_date=start_date, end_date=end_date)
        conflict_data = self._get_acled_data(all_countries, start_date=start_date, end_date=end_date)
        
        # Assess risk for each country
        risk_assessments = {}
        for country in all_countries:
            # Get country info from knowledge base
            country_info = self.knowledge_base.get("countries", {}).get(country, {})
            
            # Get baseline stability
            baseline_stability = country_info.get("stability_index", 0.7)
            
            # Filter event data for this country
            country_events = event_data[event_data["country"] == country] if not event_data.empty else pd.DataFrame()
            country_conflicts = conflict_data[conflict_data["country"] == country] if not conflict_data.empty else pd.DataFrame()
            
            # Calculate event intensity
            event_intensity = 0
            if not country_events.empty and "intensity" in country_events.columns:
                event_intensity = country_events["intensity"].mean()
            
            # Calculate conflict intensity
            conflict_intensity = 0
            if not country_conflicts.empty and "fatalities" in country_conflicts.columns:
                conflict_intensity = min(1.0, country_conflicts["fatalities"].sum() / 100)
            
            # Check if country is in conflict zones
            in_conflict_zone = False
            conflict_zone_intensity = 0
            for zone, zone_info in self.knowledge_base.get("conflict_zones", {}).items():
                if country in zone_info.get("countries_involved", []):
                    in_conflict_zone = True
                    conflict_zone_intensity = zone_info.get("intensity", 0)
                    break
            
            # Check international relations
            relations_risk = 0
            relation_count = 0
            for relation, relation_info in self.knowledge_base.get("international_relations", {}).items():
                if country in relation.split("_"):
                    relations_risk += relation_info.get("conflict_level", 0)
                    relation_count += 1
            
            if relation_count > 0:
                relations_risk /= relation_count
            
            # Calculate overall risk
            risk_factors = {
                "baseline_stability": 1.0 - baseline_stability,
                "event_intensity": event_intensity,
                "conflict_intensity": conflict_intensity,
                "conflict_zone": conflict_zone_intensity if in_conflict_zone else 0,
                "international_relations": relations_risk
            }
            
            # Weight factors based on time horizon
            if time_horizon == "short":
                weights = {
                    "baseline_stability": 0.1,
                    "event_intensity": 0.3,
                    "conflict_intensity": 0.4,
                    "conflict_zone": 0.1,
                    "international_relations": 0.1
                }
            elif time_horizon == "medium":
                weights = {
                    "baseline_stability": 0.2,
                    "event_intensity": 0.2,
                    "conflict_intensity": 0.3,
                    "conflict_zone": 0.1,
                    "international_relations": 0.2
                }
            else:  # long
                weights = {
                    "baseline_stability": 0.3,
                    "event_intensity": 0.1,
                    "conflict_intensity": 0.2,
                    "conflict_zone": 0.1,
                    "international_relations": 0.3
                }
            
            # Calculate weighted risk
            overall_risk = sum(risk_factors[factor] * weights[factor] for factor in risk_factors)
            
            # Determine risk level
            risk_level = "low"
            if overall_risk > 0.7:
                risk_level = "high"
            elif overall_risk > 0.4:
                risk_level = "medium"
            
            # Identify key risk factors
            key_risk_factors = []
            for factor, value in risk_factors.items():
                if value > 0.5:
                    key_risk_factors.append(factor)
            
            # Store assessment
            risk_assessments[country] = {
                "overall_risk": float(overall_risk),
                "risk_level": risk_level,
                "risk_factors": {k: float(v) for k, v in risk_factors.items()},
                "key_risk_factors": key_risk_factors
            }
        
        # Assess risk for each region
        region_assessments = {}
        for region in regions:
            region_info = self.knowledge_base.get("regions", {}).get(region, {})
            region_countries = region_info.get("countries", [])
            
            if not region_countries:
                continue
            
            # Calculate average risk for countries in the region
            region_risk = 0
            for country in region_countries:
                if country in risk_assessments:
                    region_risk += risk_assessments[country]["overall_risk"]
            
            region_risk /= len(region_countries)
            
            # Determine risk level
            risk_level = "low"
            if region_risk > 0.7:
                risk_level = "high"
            elif region_risk > 0.4:
                risk_level = "medium"
            
            # Store assessment
            region_assessments[region] = {
                "overall_risk": float(region_risk),
                "risk_level": risk_level,
                "countries": region_countries,
                "key_issues": region_info.get("key_issues", [])
            }
        
        # Compile results
        return {
            "country_assessments": risk_assessments,
            "region_assessments": region_assessments,
            "time_horizon": time_horizon,
            "timestamp": time.time()
        }
    
    def _analyze_regional_stability(self, input_data: Dict) -> Dict:
        """
        Analyze stability for a specific region.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Regional stability analysis
        """
        logger.info("Analyzing regional stability")
        
        # Get parameters
        region = input_data.get("region")
        time_period = input_data.get("time_period", "recent")  # recent, historical, projected
        
        # Validate input
        if not region:
            error_msg = "No region specified"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get region info from knowledge base
        region_info = self.knowledge_base.get("regions", {}).get(region, {})
        if not region_info:
            error_msg = f"Unknown region: {region}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get countries in the region
        countries = region_info.get("countries", [])
        if not countries:
            error_msg = f"No countries found for region: {region}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Set time period for data
        start_date = None
        end_date = None
        
        if time_period == "recent":
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        elif time_period == "historical":
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        else:  # projected
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        
        # Get event data for stability analysis
        event_data = self._get_gdelt_data(countries, start_date=start_date, end_date=end_date)
        conflict_data = self._get_acled_data(countries, start_date=start_date, end_date=end_date)
        
        # Analyze stability for each country
        country_stability = {}
        for country in countries:
            # Get country info from knowledge base
            country_info = self.knowledge_base.get("countries", {}).get(country, {})
            
            # Get baseline stability
            baseline_stability = country_info.get("stability_index", 0.7)
            
            # Filter event data for this country
            country_events = event_data[event_data["country"] == country] if not event_data.empty else pd.DataFrame()
            country_conflicts = conflict_data[conflict_data["country"] == country] if not conflict_data.empty else pd.DataFrame()
            
            # Calculate event trends
            event_trend = "stable"
            if not country_events.empty and len(country_events) >= 10:
                # Group by week and count events
                weekly_events = country_events.resample("W").size()
                
                if len(weekly_events) >= 2:
                    # Calculate slope of trend
                    x = np.arange(len(weekly_events))
                    slope, _, _, _, _ = np.polyfit(x, weekly_events.values, 1, full=True)
                    
                    # Determine trend direction
                    if slope > 0.1 * np.mean(weekly_events):
                        event_trend = "increasing"
                    elif slope < -0.1 * np.mean(weekly_events):
                        event_trend = "decreasing"
            
            # Calculate conflict trends
            conflict_trend = "stable"
            if not country_conflicts.empty and len(country_conflicts) >= 5:
                # Group by week and sum fatalities
                weekly_fatalities = country_conflicts.resample("W")["fatalities"].sum()
                
                if len(weekly_fatalities) >= 2:
                    # Calculate slope of trend
                    x = np.arange(len(weekly_fatalities))
                    slope, _, _, _, _ = np.polyfit(x, weekly_fatalities.values, 1, full=True)
                    
                    # Determine trend direction
                    if slope > 0.1 * np.mean(weekly_fatalities):
                        conflict_trend = "worsening"
                    elif slope < -0.1 * np.mean(weekly_fatalities):
                        conflict_trend = "improving"
            
            # Calculate stability score
            stability_factors = {
                "baseline_stability": baseline_stability,
                "event_intensity": 1.0 - (country_events["intensity"].mean() if not country_events.empty and "intensity" in country_events.columns else 0),
                "conflict_intensity": 1.0 - min(1.0, (country_conflicts["fatalities"].sum() / 100 if not country_conflicts.empty and "fatalities" in country_conflicts.columns else 0))
            }
            
            # Weight factors
            weights = {
                "baseline_stability": 0.4,
                "event_intensity": 0.3,
                "conflict_intensity": 0.3
            }
            
            # Calculate weighted stability
            stability_score = sum(stability_factors[factor] * weights[factor] for factor in stability_factors)
            
            # Determine stability level
            stability_level = "stable"
            if stability_score < 0.3:
                stability_level = "unstable"
            elif stability_score < 0.6:
                stability_level = "fragile"
            
            # Store assessment
            country_stability[country] = {
                "stability_score": float(stability_score),
                "stability_level": stability_level,
                "event_trend": event_trend,
                "conflict_trend": conflict_trend,
                "internal_issues": country_info.get("internal_issues", [])
            }
        
        # Calculate regional stability metrics
        avg_stability = sum(c["stability_score"] for c in country_stability.values()) / len(country_stability)
        
        # Count countries by stability level
        stability_counts = {
            "stable": sum(1 for c in country_stability.values() if c["stability_level"] == "stable"),
            "fragile": sum(1 for c in country_stability.values() if c["stability_level"] == "fragile"),
            "unstable": sum(1 for c in country_stability.values() if c["stability_level"] == "unstable")
        }
        
        # Determine regional stability trend
        regional_trend = "stable"
        improving_count = sum(1 for c in country_stability.values() if c["conflict_trend"] == "improving")
        worsening_count = sum(1 for c in country_stability.values() if c["conflict_trend"] == "worsening")
        
        if worsening_count > improving_count and worsening_count > len(country_stability) / 3:
            regional_trend = "deteriorating"
        elif improving_count > worsening_count and improving_count > len(country_stability) / 3:
            regional_trend = "improving"
        
        # Identify key stability factors
        key_stability_factors = []
        
        # Check for regional organizations
        if region_info.get("regional_organizations"):
            key_stability_factors.append("regional_integration")
        
        # Check for external powers involvement
        external_powers_involved = False
        for zone, zone_info in self.knowledge_base.get("conflict_zones", {}).items():
            zone_countries = zone_info.get("countries_involved", [])
            if any(country in zone_countries for country in countries):
                external_powers = [c for c in zone_info.get("international_involvement", []) if c not in countries]
                if external_powers:
                    external_powers_involved = True
                    break
        
        if external_powers_involved:
            key_stability_factors.append("external_powers_involvement")
        
        # Check for economic interdependence
        # (In a real implementation, this would use economic data)
        key_stability_factors.append("economic_interdependence")
        
        # Compile results
        return {
            "region": region,
            "regional_stability_score": float(avg_stability),
            "regional_trend": regional_trend,
            "stability_distribution": stability_counts,
            "country_stability": country_stability,
            "key_stability_factors": key_stability_factors,
            "key_regional_issues": region_info.get("key_issues", []),
            "time_period": time_period,
            "timestamp": time.time()
        }
    
    def _evaluate_conflict_potential(self, input_data: Dict) -> Dict:
        """
        Evaluate potential for conflict between countries or within a country.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Conflict potential evaluation
        """
        logger.info("Evaluating conflict potential")
        
        # Get parameters
        countries = input_data.get("countries", [])
        internal = input_data.get("internal", False)
        time_horizon = input_data.get("time_horizon", "medium")  # short, medium, long
        
        # Validate input
        if not countries:
            error_msg = "No countries specified"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Set time period for data
        start_date = None
        end_date = None
        
        if time_horizon == "short":
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            forecast_months = 3
        elif time_horizon == "medium":
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            forecast_months = 12
        else:  # long
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            forecast_months = 36
        
        # Get event data for conflict analysis
        event_data = self._get_gdelt_data(countries, start_date=start_date, end_date=end_date)
        conflict_data = self._get_acled_data(countries, start_date=start_date, end_date=end_date)
        
        if internal:
            # Evaluate internal conflict potential for each country
            conflict_potential = {}
            for country in countries:
                # Get country info from knowledge base
                country_info = self.knowledge_base.get("countries", {}).get(country, {})
                
                # Get baseline stability
                baseline_stability = country_info.get("stability_index", 0.7)
                
                # Filter event data for this country
                country_events = event_data[event_data["country"] == country] if not event_data.empty else pd.DataFrame()
                country_conflicts = conflict_data[conflict_data["country"] == country] if not conflict_data.empty else pd.DataFrame()
                
                # Calculate conflict indicators
                conflict_indicators = {
                    "baseline_instability": 1.0 - baseline_stability,
                    "recent_conflict_intensity": min(1.0, (country_conflicts["fatalities"].sum() / 50 if not country_conflicts.empty and "fatalities" in country_conflicts.columns else 0)),
                    "protest_activity": min(1.0, (country_events[country_events["event_type"].str.contains("Protest|Riot")].shape[0] / 10 if not country_events.empty else 0)),
                    "internal_issues": min(1.0, len(country_info.get("internal_issues", [])) / 5)
                }
                
                # Weight indicators based on time horizon
                if time_horizon == "short":
                    weights = {
                        "baseline_instability": 0.2,
                        "recent_conflict_intensity": 0.5,
                        "protest_activity": 0.2,
                        "internal_issues": 0.1
                    }
                elif time_horizon == "medium":
                    weights = {
                        "baseline_instability": 0.3,
                        "recent_conflict_intensity": 0.3,
                        "protest_activity": 0.2,
                        "internal_issues": 0.2
                    }
                else:  # long
                    weights = {
                        "baseline_instability": 0.4,
                        "recent_conflict_intensity": 0.2,
                        "protest_activity": 0.1,
                        "internal_issues": 0.3
                    }
                
                # Calculate weighted conflict potential
                potential_score = sum(conflict_indicators[indicator] * weights[indicator] for indicator in conflict_indicators)
                
                # Determine potential level
                potential_level = "low"
                if potential_score > 0.7:
                    potential_level = "high"
                elif potential_score > 0.4:
                    potential_level = "medium"
                
                # Identify conflict drivers
                conflict_drivers = []
                for indicator, value in conflict_indicators.items():
                    if value > 0.6:
                        conflict_drivers.append(indicator)
                
                # Add specific internal issues if available
                if "internal_issues" in conflict_drivers:
                    conflict_drivers.remove("internal_issues")
                    conflict_drivers.extend(country_info.get("internal_issues", []))
                
                # Store evaluation
                conflict_potential[country] = {
                    "potential_score": float(potential_score),
                    "potential_level": potential_level,
                    "conflict_drivers": conflict_drivers,
                    "forecast_months": forecast_months
                }
            
            # Compile results for internal conflict
            return {
                "conflict_type": "internal",
                "country_evaluations": conflict_potential,
                "time_horizon": time_horizon,
                "timestamp": time.time()
            }
            
        else:
            # Evaluate interstate conflict potential
            if len(countries) != 2:
                error_msg = "Interstate conflict evaluation requires exactly 2 countries"
                logger.error(error_msg)
                return {"error": error_msg}
            
            country1, country2 = countries
            
            # Get country info from knowledge base
            country1_info = self.knowledge_base.get("countries", {}).get(country1, {})
            country2_info = self.knowledge_base.get("countries", {}).get(country2, {})
            
            # Check existing relationship
            relationship_key = f"{country1.lower()}_{country2.lower()}"
            alt_relationship_key = f"{country2.lower()}_{country1.lower()}"
            
            relationship_info = self.knowledge_base.get("international_relations", {}).get(relationship_key, 
                                self.knowledge_base.get("international_relations", {}).get(alt_relationship_key, {}))
            
            # Calculate conflict indicators
            conflict_indicators = {
                "existing_conflict_level": relationship_info.get("conflict_level", 0.5),
                "relationship_trend": 0.7 if relationship_info.get("trend") in ["deteriorating", "stable_negative"] else 0.3,
                "power_disparity": abs(country1_info.get("military_power", 0.5) - country2_info.get("military_power", 0.5)),
                "territorial_disputes": 0.8 if any("territorial" in issue.lower() for issue in relationship_info.get("key_issues", [])) else 0.2,
                "rival_status": 0.9 if country1 in country2_info.get("key_rivals", []) or country2 in country1_info.get("key_rivals", []) else 0.1
            }
            
            # Weight indicators based on time horizon
            if time_horizon == "short":
                weights = {
                    "existing_conflict_level": 0.4,
                    "relationship_trend": 0.3,
                    "power_disparity": 0.1,
                    "territorial_disputes": 0.1,
                    "rival_status": 0.1
                }
            elif time_horizon == "medium":
                weights = {
                    "existing_conflict_level": 0.3,
                    "relationship_trend": 0.2,
                    "power_disparity": 0.2,
                    "territorial_disputes": 0.2,
                    "rival_status": 0.1
                }
            else:  # long
                weights = {
                    "existing_conflict_level": 0.2,
                    "relationship_trend": 0.1,
                    "power_disparity": 0.2,
                    "territorial_disputes": 0.3,
                    "rival_status": 0.2
                }
            
            # Calculate weighted conflict potential
            potential_score = sum(conflict_indicators[indicator] * weights[indicator] for indicator in conflict_indicators)
            
            # Determine potential level
            potential_level = "low"
            if potential_score > 0.7:
                potential_level = "high"
            elif potential_score > 0.4:
                potential_level = "medium"
            
            # Identify conflict drivers
            conflict_drivers = []
            for indicator, value in conflict_indicators.items():
                if value > 0.6 and weights[indicator] > 0.1:
                    conflict_drivers.append(indicator)
            
            # Add specific issues if available
            if relationship_info.get("key_issues"):
                conflict_drivers.extend(relationship_info.get("key_issues"))
            
            # Compile results for interstate conflict
            return {
                "conflict_type": "interstate",
                "countries": countries,
                "potential_score": float(potential_score),
                "potential_level": potential_level,
                "conflict_drivers": conflict_drivers,
                "relationship_type": relationship_info.get("relationship_type", "unknown"),
                "forecast_months": forecast_months,
                "time_horizon": time_horizon,
                "timestamp": time.time()
            }
    
    def _model_international_relations(self, input_data: Dict) -> Dict:
        """
        Model international relations between countries.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            International relations model
        """
        logger.info("Modeling international relations")
        
        # Get parameters
        countries = input_data.get("countries", [])
        include_relations = input_data.get("include_relations", True)
        include_power = input_data.get("include_power", True)
        
        # Validate input
        if not countries:
            error_msg = "No countries specified"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get country data from knowledge base
        country_data = {}
        for country in countries:
            country_info = self.knowledge_base.get("countries", {}).get(country, {})
            if country_info:
                country_data[country] = {
                    "region": country_info.get("region", "unknown"),
                    "government_type": country_info.get("government_type", "unknown"),
                    "stability_index": country_info.get("stability_index", 0.5),
                    "military_power": country_info.get("military_power", 0.5),
                    "economic_power": country_info.get("economic_power", 0.5),
                    "diplomatic_influence": country_info.get("diplomatic_influence", 0.5),
                    "key_allies": country_info.get("key_allies", []),
                    "key_rivals": country_info.get("key_rivals", [])
                }
        
        # Model relations between countries
        relations = {}
        if include_relations:
            for i, country1 in enumerate(countries):
                for country2 in countries[i+1:]:
                    # Check if relation exists in knowledge base
                    relation_key = f"{country1.lower()}_{country2.lower()}"
                    alt_relation_key = f"{country2.lower()}_{country1.lower()}"
                    
                    relation_info = self.knowledge_base.get("international_relations", {}).get(relation_key, 
                                    self.knowledge_base.get("international_relations", {}).get(alt_relation_key, {}))
                    
                    if relation_info:
                        # Use existing relation
                        relations[f"{country1}_{country2}"] = {
                            "relationship_type": relation_info.get("relationship_type", "unknown"),
                            "cooperation_level": relation_info.get("cooperation_level", 0.5),
                            "conflict_level": relation_info.get("conflict_level", 0.5),
                            "key_issues": relation_info.get("key_issues", []),
                            "trend": relation_info.get("trend", "stable")
                        }
                    else:
                        # Infer relation
                        country1_info = country_data.get(country1, {})
                        country2_info = country_data.get(country2, {})
                        
                        # Check if allies or rivals
                        is_ally = country1 in country2_info.get("key_allies", []) or country2 in country1_info.get("key_allies", [])
                        is_rival = country1 in country2_info.get("key_rivals", []) or country2 in country1_info.get("key_rivals", [])
                        
                        # Determine relationship type
                        if is_ally:
                            relationship_type = "alliance"
                            cooperation_level = 0.8
                            conflict_level = 0.2
                            trend = "stable_positive"
                        elif is_rival:
                            relationship_type = "rivalry"
                            cooperation_level = 0.2
                            conflict_level = 0.8
                            trend = "stable_negative"
                        else:
                            # Check if same region
                            same_region = country1_info.get("region") == country2_info.get("region")
                            
                            if same_region:
                                relationship_type = "regional_partners"
                                cooperation_level = 0.6
                                conflict_level = 0.4
                                trend = "stable"
                            else:
                                relationship_type = "neutral"
                                cooperation_level = 0.5
                                conflict_level = 0.5
                                trend = "stable"
                        
                        relations[f"{country1}_{country2}"] = {
                            "relationship_type": relationship_type,
                            "cooperation_level": cooperation_level,
                            "conflict_level": conflict_level,
                            "key_issues": [],
                            "trend": trend,
                            "inferred": True
                        }
        
        # Calculate power metrics
        power_metrics = {}
        if include_power:
            for country in countries:
                country_info = country_data.get(country, {})
                
                # Calculate composite power index
                military = country_info.get("military_power", 0.5)
                economic = country_info.get("economic_power", 0.5)
                diplomatic = country_info.get("diplomatic_influence", 0.5)
                
                composite_power = 0.4 * military + 0.4 * economic + 0.2 * diplomatic
                
                # Count allies and rivals
                num_allies = len(country_info.get("key_allies", []))
                num_rivals = len(country_info.get("key_rivals", []))
                
                # Calculate alliance power
                alliance_power = 0
                for ally in country_info.get("key_allies", []):
                    ally_info = self.knowledge_base.get("countries", {}).get(ally, {})
                    ally_power = 0.4 * ally_info.get("military_power", 0.5) + 0.4 * ally_info.get("economic_power", 0.5) + 0.2 * ally_info.get("diplomatic_influence", 0.5)
                    alliance_power += ally_power
                
                # Store power metrics
                power_metrics[country] = {
                    "composite_power": float(composite_power),
                    "military_power": float(military),
                    "economic_power": float(economic),
                    "diplomatic_influence": float(diplomatic),
                    "num_allies": num_allies,
                    "num_rivals": num_rivals,
                    "alliance_power": float(alliance_power)
                }
        
        # Compile results
        return {
            "countries": list(country_data.keys()),
            "country_data": country_data,
            "relations": relations,
            "power_metrics": power_metrics,
            "timestamp": time.time()
        }
    
    def _analyze_political_trends(self, input_data: Dict) -> Dict:
        """
        Analyze political trends for countries or regions.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Political trend analysis
        """
        logger.info("Analyzing political trends")
        
        # Get parameters
        countries = input_data.get("countries", [])
        regions = input_data.get("regions", [])
        trend_types = input_data.get("trend_types", ["governance", "stability", "international_position"])
        
        # Validate input
        if not countries and not regions:
            error_msg = "No countries or regions specified"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Expand regions to countries
        all_countries = list(countries)
        for region in regions:
            region_info = self.knowledge_base.get("regions", {}).get(region, {})
            region_countries = region_info.get("countries", [])
            all_countries.extend(region_countries)
        
        # Remove duplicates
        all_countries = list(set(all_countries))
        
        # Get event data for trend analysis
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        event_data = self._get_gdelt_data(all_countries, start_date=start_date, end_date=end_date)
        
        # Analyze trends for each country
        country_trends = {}
        for country in all_countries:
            # Get country info from knowledge base
            country_info = self.knowledge_base.get("countries", {}).get(country, {})
            
            # Filter event data for this country
            country_events = event_data[event_data["country"] == country] if not event_data.empty else pd.DataFrame()
            
            # Initialize trends
            trends = {}
            
            # Analyze governance trends
            if "governance" in trend_types:
                # In a real implementation, this would use more sophisticated analysis
                # For now, use a simple model based on knowledge base and events
                
                government_type = country_info.get("government_type", "unknown")
                stability_index = country_info.get("stability_index", 0.5)
                
                # Count protest events
                protest_count = country_events[country_events["event_type"].str.contains("Protest|Riot")].shape[0] if not country_events.empty else 0
                
                # Determine governance trend
                if government_type in ["parliamentary_democracy", "federal_republic"]:
                    if stability_index > 0.7:
                        governance_trend = "stable_democracy"
                    elif protest_count > 10:
                        governance_trend = "contested_democracy"
                    else:
                        governance_trend = "functioning_democracy"
                elif government_type in ["semi_authoritarian", "one_party_state"]:
                    if stability_index > 0.7:
                        governance_trend = "stable_authoritarianism"
                    elif protest_count > 10:
                        governance_trend = "contested_authoritarianism"
                    else:
                        governance_trend = "consolidated_authoritarianism"
                else:
                    governance_trend = "mixed_governance"
                
                trends["governance"] = {
                    "current_type": government_type,
                    "trend": governance_trend,
                    "protest_activity": protest_count,
                    "key_issues": country_info.get("internal_issues", [])
                }
            
            # Analyze stability trends
            if "stability" in trend_types:
                # Get baseline stability
                stability_index = country_info.get("stability_index", 0.5)
                
                # Check if country is in conflict zones
                in_conflict_zone = False
                conflict_intensity = 0
                for zone, zone_info in self.knowledge_base.get("conflict_zones", {}).items():
                    if country in zone_info.get("countries_involved", []):
                        in_conflict_zone = True
                        conflict_intensity = zone_info.get("intensity", 0)
                        break
                
                # Determine stability trend
                if in_conflict_zone:
                    if conflict_intensity > 0.7:
                        stability_trend = "active_conflict"
                    else:
                        stability_trend = "fragile_peace"
                elif stability_index > 0.8:
                    stability_trend = "stable"
                elif stability_index > 0.6:
                    stability_trend = "mostly_stable"
                elif stability_index > 0.4:
                    stability_trend = "fragile"
                else:
                    stability_trend = "unstable"
                
                trends["stability"] = {
                    "stability_index": float(stability_index),
                    "trend": stability_trend,
                    "in_conflict_zone": in_conflict_zone,
                    "conflict_intensity": float(conflict_intensity) if in_conflict_zone else 0
                }
            
            # Analyze international position trends
            if "international_position" in trend_types:
                # Get power metrics
                military_power = country_info.get("military_power", 0.5)
                economic_power = country_info.get("economic_power", 0.5)
                diplomatic_influence = country_info.get("diplomatic_influence", 0.5)
                
                # Count allies and rivals
                num_allies = len(country_info.get("key_allies", []))
                num_rivals = len(country_info.get("key_rivals", []))
                
                # Determine international position
                composite_power = 0.4 * military_power + 0.4 * economic_power + 0.2 * diplomatic_influence
                
                if composite_power > 0.8:
                    position = "global_power"
                elif composite_power > 0.6:
                    position = "major_power"
                elif composite_power > 0.4:
                    position = "regional_power"
                else:
                    position = "minor_power"
                
                # Determine alignment
                if num_allies > num_rivals * 2:
                    alignment = "well_aligned"
                elif num_allies > num_rivals:
                    alignment = "moderately_aligned"
                elif num_rivals > num_allies * 2:
                    alignment = "isolated"
                else:
                    alignment = "contested"
                
                trends["international_position"] = {
                    "position": position,
                    "alignment": alignment,
                    "composite_power": float(composite_power),
                    "num_allies": num_allies,
                    "num_rivals": num_rivals,
                    "key_allies": country_info.get("key_allies", []),
                    "key_rivals": country_info.get("key_rivals", [])
                }
            
            # Store country trends
            country_trends[country] = trends
        
        # Analyze regional trends
        region_trends = {}
        for region in regions:
            region_info = self.knowledge_base.get("regions", {}).get(region, {})
            region_countries = region_info.get("countries", [])
            
            if not region_countries:
                continue
            
            # Initialize trends
            trends = {}
            
            # Analyze governance trends
            if "governance" in trend_types:
                # Count countries by governance type
                democracy_count = 0
                authoritarian_count = 0
                mixed_count = 0
                
                for country in region_countries:
                    if country in country_trends:
                        gov_trend = country_trends[country].get("governance", {}).get("trend", "")
                        if "democracy" in gov_trend:
                            democracy_count += 1
                        elif "authoritarianism" in gov_trend:
                            authoritarian_count += 1
                        else:
                            mixed_count += 1
                
                # Determine regional governance trend
                if democracy_count > authoritarian_count * 2:
                    governance_trend = "democratic_consolidation"
                elif democracy_count > authoritarian_count:
                    governance_trend = "democratic_leaning"
                elif authoritarian_count > democracy_count * 2:
                    governance_trend = "authoritarian_consolidation"
                elif authoritarian_count > democracy_count:
                    governance_trend = "authoritarian_leaning"
                else:
                    governance_trend = "mixed_governance"
                
                trends["governance"] = {
                    "trend": governance_trend,
                    "democracy_count": democracy_count,
                    "authoritarian_count": authoritarian_count,
                    "mixed_count": mixed_count
                }
            
            # Analyze stability trends
            if "stability" in trend_types:
                # Calculate average stability
                stability_values = [country_trends[country].get("stability", {}).get("stability_index", 0.5) 
                                   for country in region_countries if country in country_trends]
                
                avg_stability = sum(stability_values) / len(stability_values) if stability_values else 0.5
                
                # Count countries in conflict
                conflict_count = sum(1 for country in region_countries 
                                    if country in country_trends and 
                                    country_trends[country].get("stability", {}).get("in_conflict_zone", False))
                
                # Determine regional stability trend
                if avg_stability > 0.7 and conflict_count == 0:
                    stability_trend = "stable_region"
                elif avg_stability > 0.6 and conflict_count <= 1:
                    stability_trend = "mostly_stable_region"
                elif conflict_count > len(region_countries) / 3:
                    stability_trend = "conflict_prone_region"
                else:
                    stability_trend = "mixed_stability_region"
                
                trends["stability"] = {
                    "trend": stability_trend,
                    "average_stability": float(avg_stability),
                    "conflict_count": conflict_count,
                    "total_countries": len(region_countries)
                }
            
            # Analyze international position trends
            if "international_position" in trend_types:
                # Identify major powers in the region
                major_powers = [country for country in region_countries 
                               if country in country_trends and 
                               country_trends[country].get("international_position", {}).get("position") in ["global_power", "major_power"]]
                
                # Check for external influence
                external_influence = []
                for relation, relation_info in self.knowledge_base.get("international_relations", {}).items():
                    countries_in_relation = relation.split("_")
                    if any(country in region_countries for country in countries_in_relation) and not all(country in region_countries for country in countries_in_relation):
                        external_country = [c for c in countries_in_relation if c not in region_countries][0]
                        external_info = self.knowledge_base.get("countries", {}).get(external_country, {})
                        if external_info.get("composite_power", 0) > 0.6:
                            external_influence.append(external_country)
                
                # Determine regional position trend
                if len(major_powers) > 1:
                    position_trend = "multipolar_region"
                elif len(major_powers) == 1:
                    position_trend = "dominant_power_region"
                elif external_influence:
                    position_trend = "externally_influenced_region"
                else:
                    position_trend = "balanced_region"
                
                trends["international_position"] = {
                    "trend": position_trend,
                    "major_powers": major_powers,
                    "external_influence": list(set(external_influence))
                }
            
            # Store region trends
            region_trends[region] = trends
        
        # Compile results
        return {
            "country_trends": country_trends,
            "region_trends": region_trends,
            "trend_types": trend_types,
            "timestamp": time.time()
        }
    
    def _get_event_data(self, input_data: Dict) -> Dict:
        """
        Get raw event data for countries.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Event data
        """
        logger.info("Getting event data")
        
        # Get parameters
        countries = input_data.get("countries", [])
        event_types = input_data.get("event_types")
        start_date = input_data.get("start_date")
        end_date = input_data.get("end_date")
        data_source = input_data.get("data_source", "gdelt")  # gdelt or acled
        
        # Validate input
        if not countries:
            error_msg = "No countries specified"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get event data
        if data_source.lower() == "acled":
            df = self._get_acled_data(countries, event_types, start_date, end_date)
        else:
            df = self._get_gdelt_data(countries, event_types, start_date, end_date)
        
        # Convert to CSV format
        csv_data = df.to_csv() if not df.empty else ""
        
        # Compile results
        return {
            "data": csv_data,
            "countries": countries,
            "event_types": event_types,
            "start_date": start_date,
            "end_date": end_date,
            "data_source": data_source,
            "timestamp": time.time()
        }
    
    def _get_country_profile(self, input_data: Dict) -> Dict:
        """
        Get comprehensive profile for a country.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Country profile
        """
        logger.info("Getting country profile")
        
        # Get parameters
        country = input_data.get("country")
        
        # Validate input
        if not country:
            error_msg = "No country specified"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get country info from knowledge base
        country_info = self.knowledge_base.get("countries", {}).get(country, {})
        if not country_info:
            error_msg = f"Unknown country: {country}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get region info
        region = country_info.get("region", "unknown")
        region_info = self.knowledge_base.get("regions", {}).get(region, {})
        
        # Get recent events
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        event_data = self._get_gdelt_data([country], start_date=start_date, end_date=end_date)
        conflict_data = self._get_acled_data([country], start_date=start_date, end_date=end_date)
        
        # Count events by type
        event_counts = {}
        if not event_data.empty and "event_type" in event_data.columns:
            event_counts = event_data["event_type"].value_counts().to_dict()
        
        # Count conflicts by type
        conflict_counts = {}
        if not conflict_data.empty and "event_type" in conflict_data.columns:
            conflict_counts = conflict_data["event_type"].value_counts().to_dict()
        
        # Get international relations
        relations = {}
        for relation, relation_info in self.knowledge_base.get("international_relations", {}).items():
            countries_in_relation = relation.split("_")
            if country.lower() in countries_in_relation:
                other_country = [c for c in countries_in_relation if c.lower() != country.lower()][0]
                relations[other_country] = {
                    "relationship_type": relation_info.get("relationship_type", "unknown"),
                    "cooperation_level": relation_info.get("cooperation_level", 0.5),
                    "conflict_level": relation_info.get("conflict_level", 0.5),
                    "key_issues": relation_info.get("key_issues", []),
                    "trend": relation_info.get("trend", "stable")
                }
        
        # Check if country is in conflict zones
        conflict_zones = []
        for zone, zone_info in self.knowledge_base.get("conflict_zones", {}).items():
            if country in zone_info.get("countries_involved", []):
                conflict_zones.append({
                    "name": zone,
                    "intensity": zone_info.get("intensity", 0),
                    "duration": zone_info.get("duration", "unknown"),
                    "type": zone_info.get("type", "unknown"),
                    "countries_involved": zone_info.get("countries_involved", []),
                    "resolution_prospects": zone_info.get("resolution_prospects", 0)
                })
        
        # Compile profile
        profile = {
            "country": country,
            "region": region,
            "government_type": country_info.get("government_type", "unknown"),
            "stability_index": country_info.get("stability_index", 0.5),
            "power_metrics": {
                "military_power": country_info.get("military_power", 0.5),
                "economic_power": country_info.get("economic_power", 0.5),
                "diplomatic_influence": country_info.get("diplomatic_influence", 0.5)
            },
            "key_allies": country_info.get("key_allies", []),
            "key_rivals": country_info.get("key_rivals", []),
            "internal_issues": country_info.get("internal_issues", []),
            "regional_context": {
                "region_name": region,
                "regional_organizations": region_info.get("regional_organizations", []),
                "key_regional_issues": region_info.get("key_issues", []),
                "regional_stability": region_info.get("stability_index", 0.5)
            },
            "recent_events": {
                "event_counts": event_counts,
                "conflict_counts": conflict_counts,
                "period": f"{start_date} to {end_date}"
            },
            "international_relations": relations,
            "conflict_involvement": conflict_zones
        }
        
        # Compile results
        return {
            "profile": profile,
            "timestamp": time.time()
        }
    
    def _visualize_geopolitical_data(self, input_data: Dict) -> Dict:
        """
        Visualize geopolitical data.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Visualization data
        """
        logger.info("Visualizing geopolitical data")
        
        # Get parameters
        countries = input_data.get("countries", [])
        visualization_type = input_data.get("visualization_type", "event_timeline")
        start_date = input_data.get("start_date")
        end_date = input_data.get("end_date")
        
        # Validate input
        if not countries:
            error_msg = "No countries specified"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if not start_date:
            start_date_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=180)  # 180 days
            start_date = start_date_dt.strftime("%Y-%m-%d")
        
        # Get data for visualization
        event_data = self._get_gdelt_data(countries, start_date=start_date, end_date=end_date)
        
        # Create visualization
        visualization = {}
        
        try:
            # Set up figure
            plt.figure(figsize=(12, 8))
            
            if visualization_type == "event_timeline":
                # Create event timeline
                if not event_data.empty:
                    # Group by date and country
                    event_counts = event_data.groupby([pd.Grouper(freq="W"), "country"]).size().unstack(fill_value=0)
                    
                    # Plot timeline
                    for country in event_counts.columns:
                        plt.plot(event_counts.index, event_counts[country], marker='o', linestyle='-', label=country)
                    
                    # Add title and labels
                    plt.title("Event Timeline by Country")
                    plt.xlabel("Date")
                    plt.ylabel("Number of Events")
                    plt.legend()
                    
                else:
                    plt.text(0.5, 0.5, "No event data available", horizontalalignment='center', verticalalignment='center')
                
            elif visualization_type == "event_distribution":
                # Create event distribution
                if not event_data.empty and "event_type" in event_data.columns:
                    # Group by event type and country
                    event_counts = event_data.groupby(["event_type", "country"]).size().unstack(fill_value=0)
                    
                    # Plot distribution
                    event_counts.plot(kind="bar", figsize=(12, 8))
                    
                    # Add title and labels
                    plt.title("Event Distribution by Type and Country")
                    plt.xlabel("Event Type")
                    plt.ylabel("Number of Events")
                    plt.legend(title="Country")
                    
                else:
                    plt.text(0.5, 0.5, "No event data available", horizontalalignment='center', verticalalignment='center')
                
            elif visualization_type == "conflict_intensity":
                # Create conflict intensity map
                if not event_data.empty and "intensity" in event_data.columns:
                    # Group by date and country
                    intensity_data = event_data.groupby([pd.Grouper(freq="W"), "country"])["intensity"].mean().unstack(fill_value=0)
                    
                    # Create heatmap
                    sns.heatmap(intensity_data.T, cmap="YlOrRd", linewidths=0.5)
                    
                    # Add title and labels
                    plt.title("Conflict Intensity by Country Over Time")
                    plt.xlabel("Date")
                    plt.ylabel("Country")
                    
                else:
                    plt.text(0.5, 0.5, "No intensity data available", horizontalalignment='center', verticalalignment='center')
                
            elif visualization_type == "power_comparison":
                # Create power comparison chart
                country_data = {}
                for country in countries:
                    country_info = self.knowledge_base.get("countries", {}).get(country, {})
                    if country_info:
                        country_data[country] = {
                            "military_power": country_info.get("military_power", 0.5),
                            "economic_power": country_info.get("economic_power", 0.5),
                            "diplomatic_influence": country_info.get("diplomatic_influence", 0.5)
                        }
                
                if country_data:
                    # Convert to DataFrame
                    power_df = pd.DataFrame(country_data).T
                    
                    # Plot radar chart
                    categories = list(power_df.columns)
                    N = len(categories)
                    
                    # Create angles for radar chart
                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]
                    
                    # Create subplot with polar projection
                    ax = plt.subplot(111, polar=True)
                    
                    # Draw one axis per variable and add labels
                    plt.xticks(angles[:-1], categories, size=8)
                    
                    # Draw ylabels
                    ax.set_rlabel_position(0)
                    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=7)
                    plt.ylim(0, 1)
                    
                    # Plot each country
                    for country in power_df.index:
                        values = power_df.loc[country].values.flatten().tolist()
                        values += values[:1]
                        ax.plot(angles, values, linewidth=1, linestyle='solid', label=country)
                        ax.fill(angles, values, alpha=0.1)
                    
                    # Add legend
                    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                    
                    # Add title
                    plt.title("Power Comparison by Country")
                    
                else:
                    plt.text(0.5, 0.5, "No power data available", horizontalalignment='center', verticalalignment='center')
                
            else:
                plt.text(0.5, 0.5, f"Unsupported visualization type: {visualization_type}", horizontalalignment='center', verticalalignment='center')
            
            # Adjust layout
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
            visualization = {
                "image_data": img_str,
                "format": "png",
                "encoding": "base64"
            }
            
            # Close figure
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            visualization = {"error": str(e)}
        
        # Compile results
        return {
            "visualization": visualization,
            "countries": countries,
            "visualization_type": visualization_type,
            "start_date": start_date,
            "end_date": end_date,
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
            "data_sources": ["GDELT", "ACLED"],
            "knowledge_base_categories": list(self.knowledge_base.keys()),
            "regions": list(self.knowledge_base.get("regions", {}).keys()),
            "countries": list(self.knowledge_base.get("countries", {}).keys())
        }
