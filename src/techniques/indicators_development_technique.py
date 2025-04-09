"""
Indicators Development Technique implementation.
This module provides the IndicatorsDevelopmentTechnique class for creating monitoring indicators.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional

from .analytical_technique import AnalyticalTechnique
from utils.llm_integration import call_llm, extract_content, parse_json_response, MODEL_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndicatorsDevelopmentTechnique(AnalyticalTechnique):
    """
    Develops indicators for monitoring key aspects of a situation or forecast.
    
    This technique identifies specific, measurable indicators that can be used
    to track the evolution of a situation, validate forecasts, and provide
    early warning of significant changes.
    """
    
    def execute(self, context, parameters):
        """
        Execute the indicators development technique.
        
        Args:
            context: The analysis context
            parameters: Parameters for the technique
            
        Returns:
            Dictionary containing indicators development results
        """
        logger.info(f"Executing IndicatorsDevelopmentTechnique for question: {context.question[:50]}...")
        
        # Get parameters or use defaults
        source_technique = parameters.get("source_technique", None)
        indicator_types = parameters.get("indicator_types", ["leading", "coincident", "lagging"])
        
        # Step 1: Identify key aspects to monitor
        key_aspects = self._identify_key_aspects(context, source_technique)
        
        # Step 2: Develop indicators for each aspect
        indicators = self._develop_indicators(context.question, key_aspects, indicator_types)
        
        # Step 3: Assess indicator quality
        indicator_assessment = self._assess_indicator_quality(indicators)
        
        # Step 4: Develop monitoring plan
        monitoring_plan = self._develop_monitoring_plan(indicators, indicator_assessment)
        
        # Step 5: Generate final synthesis
        synthesis = self._generate_synthesis(context.question, indicators, indicator_assessment, monitoring_plan)
        
        return {
            "technique": "Indicators Development",
            "status": "Completed",
            "key_aspects": key_aspects,
            "indicators": indicators,
            "indicator_assessment": indicator_assessment,
            "monitoring_plan": monitoring_plan,
            "final_judgment": synthesis.get("final_judgment", "No judgment provided"),
            "judgment_rationale": synthesis.get("judgment_rationale", "No rationale provided"),
            "confidence_level": synthesis.get("confidence_level", "Medium"),
            "potential_biases": synthesis.get("potential_biases", [])
        }
    
    def get_required_mcps(self):
        """
        Return list of MCPs that enhance this technique.
        
        Returns:
            List of MCP names that enhance this technique
        """
        return ["monitoring_mcp", "data_analysis_mcp"]
    
    def _identify_key_aspects(self, context, source_technique):
        """
        Identify key aspects of the situation to monitor.
        
        Args:
            context: The analysis context
            source_technique: Name of technique to source aspects from
            
        Returns:
            List of key aspect dictionaries
        """
        logger.info("Identifying key aspects to monitor...")
        
        # Try to get aspects from source technique if specified
        if source_technique and source_technique in context.results:
            source_results = context.results[source_technique]
            
            # Look for aspects in common fields
            potential_aspects = []
            
            # Check for key drivers
            if "key_drivers" in source_results:
                for driver in source_results["key_drivers"]:
                    potential_aspects.append({
                        "name": driver,
                        "category": "Driver",
                        "importance": "High"
                    })
            
            # Check for key insights
            if "key_insights" in source_results:
                for insight in source_results["key_insights"]:
                    potential_aspects.append({
                        "name": insight,
                        "category": "Insight",
                        "importance": "High"
                    })
            
            # Check for critical uncertainties
            if "critical_uncertainties" in source_results:
                for uncertainty in source_results["critical_uncertainties"]:
                    potential_aspects.append({
                        "name": uncertainty,
                        "category": "Uncertainty",
                        "importance": "High"
                    })
            
            # If we found potential aspects, process them
            if potential_aspects:
                logger.info(f"Found {len(potential_aspects)} potential aspects from {source_technique}")
                return self._process_aspects(potential_aspects)
        
        # Use monitoring MCP if available
        monitoring_mcp = self.mcp_registry.get_mcp("monitoring_mcp")
        
        if monitoring_mcp:
            try:
                logger.info("Using monitoring MCP")
                key_aspects = monitoring_mcp.identify_key_aspects(context.question)
                return key_aspects
            except Exception as e:
                logger.error(f"Error using monitoring MCP: {e}")
                # Fall through to LLM-based identification
        
        # Use LLM to identify key aspects
        prompt = f"""
        Identify key aspects to monitor for the following analytical question:
        
        "{context.question}"
        
        For this analysis:
        1. Identify 5-7 key aspects of the situation that should be monitored
        2. Include a mix of different types of aspects (e.g., economic, political, technological, social)
        3. Focus on aspects that are most relevant to the question and likely to change over time
        4. Provide a clear name and description for each aspect
        5. Explain why each aspect is important to monitor
        
        Return your response as a JSON object with the following structure:
        {{
            "key_aspects": [
                {{
                    "name": "Name of the aspect",
                    "description": "Brief description of the aspect",
                    "category": "Category of the aspect (e.g., Economic, Political, Technological, Social)",
                    "importance": "Explanation of why this aspect is important to monitor"
                }},
                ...
            ]
        }}
        """
        
        model_config = MODEL_CONFIG["sonar"]
        response = call_llm(prompt, model_config)
        content = extract_content(response)
        
        try:
            parsed_response = parse_json_response(content)
            
            if parsed_response.get("fallback_generated"):
                logger.error(f"Error identifying key aspects: {parsed_response.get('error')}")
                return self._generate_fallback_key_aspects(context.question)
            
            key_aspects = parsed_response.get("key_aspects", [])
            
            if not key_aspects:
                logger.warning("No key aspects identified")
                return self._generate_fallback_key_aspects(context.question)
            
            return self._process_aspects(key_aspects)
        
        except Exception as e:
            logger.error(f"Error parsing key aspects: {e}")
            return self._generate_fallback_key_aspects(context.question)
    
    def _process_aspects(self, aspects):
        """
        Process and normalize aspects.
        
        Args:
            aspects: List of raw aspect dictionaries
            
        Returns:
            List of processed aspect dictionaries
        """
        processed_aspects = []
        seen_names = set()
        
        for aspect in aspects:
            # Extract name and ensure it exists
            name = aspect.get("name", "")
            if not name:
                continue
            
            # Skip duplicates
            if name.lower() in seen_names:
                continue
            
            seen_names.add(name.lower())
            
            # Ensure all required fields exist
            processed_aspect = {
                "name": name,
                "description": aspect.get("description", f"Description of {name}"),
                "category": aspect.get("category", "General"),
                "importance": aspect.get("importance", "Medium")
            }
            
            processed_aspects.append(processed_aspect)
        
        return processed_aspects
    
    def _generate_fallback_key_aspects(self, question):
        """
        Generate fallback key aspects when identification fails.
        
        Args:
            question: The analytical question
            
        Returns:
            List of fallback key aspect dictionaries
        """
        return [
            {
                "name": "Economic Performance",
                "description": "Overall economic conditions including growth, inflation, and employment",
                "category": "Economic",
                "importance": "Economic conditions provide a foundation for many other developments and directly impact resources available"
            },
            {
                "name": "Technological Adoption",
                "description": "Rate and patterns of adoption for relevant technologies",
                "category": "Technological",
                "importance": "Technology adoption rates indicate market acceptance and potential for disruptive change"
            },
            {
                "name": "Regulatory Environment",
                "description": "Changes in laws, regulations, and policies affecting the domain",
                "category": "Political",
                "importance": "Regulatory changes can rapidly alter the operating environment and create new constraints or opportunities"
            },
            {
                "name": "Competitive Landscape",
                "description": "Changes in the number, type, and behavior of competitors",
                "category": "Market",
                "importance": "Competitive dynamics shape strategic options and influence success probabilities"
            },
            {
                "name": "Stakeholder Sentiment",
                "description": "Attitudes and perceptions of key stakeholders",
                "category": "Social",
                "importance": "Stakeholder support or opposition can accelerate or block developments"
            }
        ]
    
    def _develop_indicators(self, question, key_aspects, indicator_types):
        """
        Develop indicators for each key aspect.
        
        Args:
            question: The analytical question
            key_aspects: List of key aspect dictionaries
            indicator_types: List of indicator types to develop
            
        Returns:
            Dictionary mapping aspect names to their indicators
        """
        logger.info(f"Developing indicators for {len(key_aspects)} key aspects...")
        
        # Use monitoring MCP if available
        monitoring_mcp = self.mcp_registry.get_mcp("monitoring_mcp")
        
        if monitoring_mcp:
            try:
                logger.info("Using monitoring MCP")
                indicators = monitoring_mcp.develop_indicators(question, key_aspects, indicator_types)
                return indicators
            except Exception as e:
                logger.error(f"Error using monitoring MCP: {e}")
                # Fall through to LLM-based development
        
        indicators = {}
        
        for aspect in key_aspects:
            aspect_name = aspect.get("name", "")
            aspect_description = aspect.get("description", "")
            aspect_category = aspect.get("category", "")
            
            logger.info(f"Developing indicators for aspect: {aspect_name}")
            
            # Use LLM to develop indicators for this aspect
            prompt = f"""
            Develop monitoring indicators for the following aspect related to this question:
            
            Question: "{question}"
            
            Aspect: {aspect_name}
            Description: {aspect_description}
            Category: {aspect_category}
            
            For this aspect, develop:
            1. 2-3 leading indicators (early warning signs that precede changes)
            2. 2-3 coincident indicators (real-time measures of current state)
            3. 2-3 lagging indicators (confirmatory measures that follow changes)
            
            For each indicator:
            1. Provide a clear name and description
            2. Specify how it would be measured or observed
            3. Explain what changes in this indicator would signify
            4. Assess how reliable this indicator is likely to be
            
            Return your response as a JSON object with the following structure:
            {{
                "leading_indicators": [
                    {{
                        "name": "Name of the indicator",
                        "description": "Description of what this indicator measures",
                        "measurement_method": "How this indicator would be measured or observed",
                        "change_significance": "What changes in this indicator would signify",
                        "reliability": "High/Medium/Low"
                    }},
                    ...
                ],
                "coincident_indicators": [
                    {{
                        "name": "Name of the indicator",
                        "description": "Description of what this indicator measures",
                        "measurement_method": "How this indicator would be measured or observed",
                        "change_significance": "What changes in this indicator would signify",
                        "reliability": "High/Medium/Low"
                    }},
                    ...
                ],
                "lagging_indicators": [
                    {{
                        "name": "Name of the indicator",
                        "description": "Description of what this indicator measures",
                        "measurement_method": "How this indicator would be measured or observed",
                        "change_significance": "What changes in this indicator would signify",
                        "reliability": "High/Medium/Low"
                    }},
                    ...
                ]
            }}
            """
            
            model_config = MODEL_CONFIG["sonar"]
            response = call_llm(prompt, model_config)
            content = extract_content(response)
            
            try:
                parsed_response = parse_json_response(content)
                
                if parsed_response.get("fallback_generated"):
                    logger.error(f"Error developing indicators: {parsed_response.get('error')}")
                    indicators[aspect_name] = self._generate_fallback_indicators(aspect_name, aspect_category)
                else:
                    indicators[aspect_name] = parsed_response
            
            except Exception as e:
                logger.error(f"Error parsing indicators: {e}")
                indicators[aspect_name] = self._generate_fallback_indicators(aspect_name, aspect_category)
        
        return indicators
    
    def _generate_fallback_indicators(self, aspect_name, aspect_category):
        """
        Generate fallback indicators when development fails.
        
        Args:
            aspect_name: Name of the aspect
            aspect_category: Category of the aspect
            
        Returns:
            Dictionary containing fallback indicators
        """
        category_indicators = {
            "Economic": {
                "leading_indicators": [
                    {
                        "name": "Business Confidence Index",
                        "description": "Survey-based measure of business expectations",
                        "measurement_method": "Regular surveys of business leaders",
                        "change_significance": "Declining confidence often precedes economic slowdowns",
                        "reliability": "Medium"
                    },
                    {
                        "name": "New Business Applications",
                        "description": "Rate of new business formation",
                        "measurement_method": "Government registration data",
                        "change_significance": "Declining applications suggest reduced economic optimism",
                        "reliability": "Medium"
                    }
                ],
                "coincident_indicators": [
                    {
                        "name": "Current Economic Activity Index",
                        "description": "Composite measure of current economic performance",
                        "measurement_method": "Combination of production, employment, and sales data",
                        "change_significance": "Directly reflects current economic conditions",
                        "reliability": "High"
                    },
                    {
                        "name": "Consumer Spending",
                        "description": "Current level of consumer expenditures",
                        "measurement_method": "Retail sales data and payment processor information",
                        "change_significance": "Reflects current economic activity and consumer confidence",
                        "reliability": "High"
                    }
                ],
                "lagging_indicators": [
                    {
                        "name": "Unemployment Rate",
                        "description": "Percentage of labor force that is unemployed",
                        "measurement_method": "Government labor surveys",
                        "change_significance": "Confirms economic trends after they've occurred",
                        "reliability": "High"
                    },
                    {
                        "name": "Corporate Profits",
                        "description": "Reported profits of major corporations",
                        "measurement_method": "Quarterly financial reports",
                        "change_significance": "Confirms economic performance after the fact",
                        "reliability": "High"
                    }
                ]
            },
            "Technological": {
                "leading_indicators": [
                    {
                        "name": "R&D Investment Trends",
                        "description": "Changes in research and development spending",
                        "measurement_method": "Corporate financial reports and government data",
                        "change_significance": "Indicates future innovation potential",
                        "reliability": "Medium"
                    },
                    {
                        "name": "Patent Applications",
                        "description": "Rate of new patent filings in relevant domains",
                        "measurement_method": "Patent office data",
                        "change_significance": "Early indicator of innovation direction",
                        "reliability": "Medium"
                    }
                ],
                "coincident_indicators": [
                    {
                        "name": "Technology Adoption Rate",
                        "description": "Current rate of adoption for relevant technologies",
                        "measurement_method": "Market research surveys and sales data",
                        "change_significance": "Reflects current market acceptance",
                        "reliability": "High"
                    },
                    {
                        "name": "Industry Conference Themes",
                        "description": "Prevalent topics at major industry conferences",
                        "measurement_method": "Analysis of conference agendas and presentations",
                        "change_significance": "Indicates current industry focus",
                        "reliability": "Medium"
                    }
                ],
                "lagging_indicators": [
                    {
                        "name": "Market Share Distribution",
                        "description": "Distribution of market share among competitors",
                        "measurement_method": "Market research reports",
                        "change_significance": "Confirms which technologies have succeeded",
                        "reliability": "High"
                    },
                    {
                        "name": "Industry Standards Adoption",
                        "description": "Formal adoption of standards based on technologies",
                        "measurement_method": "Standards body publications",
                        "change_significance": "Confirms technology has become established",
                        "reliability": "High"
                    }
                ]
            },
            "Political": {
                "leading_indicators": [
                    {
                        "name": "Legislative Proposals",
                        "description": "New bills and proposals in relevant policy areas",
                        "measurement_method": "Legislative tracking services",
                        "change_significance": "Early indicator of potential regulatory changes",
                        "reliability": "Medium"
                    },
                    {
                        "name": "Regulatory Agency Staffing",
                        "description": "Changes in personnel at relevant regulatory agencies",
                        "measurement_method": "Agency announcements and public records",
                        "change_significance": "Signals potential shifts in regulatory priorities",
                        "reliability": "Medium"
                    }
                ],
                "coincident_indicators": [
                    {
                        "name": "Policy Implementation Rate",
                        "description": "Rate at which announced policies are implemented",
                        "measurement_method": "Government implementation reports",
                        "change_significance": "Reflects current regulatory environment",
                        "reliability": "High"
                    },
                    {
                        "name": "Regulatory Enforcement Actions",
                        "description": "Number and type of enforcement actions",
                        "measurement_method": "Regulatory agency reports",
                        "change_significance": "Indicates current regulatory priorities",
                        "reliability": "High"
                    }
                ],
                "lagging_indicators": [
                    {
                        "name": "Policy Impact Assessments",
                        "description": "Formal evaluations of policy effectiveness",
                        "measurement_method": "Government and independent assessment reports",
                        "change_significance": "Confirms actual impact of regulatory changes",
                        "reliability": "High"
                    },
                    {
                        "name": "Legal Precedents",
                        "description": "Court decisions establishing interpretation of regulations",
                        "measurement_method": "Legal database analysis",
                        "change_significance": "Confirms how regulations are applied in practice",
                        "reliability": "High"
                    }
                ]
            },
            "Social": {
                "leading_indicators": [
                    {
                        "name": "Social Media Sentiment",
                        "description": "Sentiment analysis of social media discussions",
                        "measurement_method": "Social media monitoring tools",
                        "change_significance": "Early indicator of shifting public opinion",
                        "reliability": "Low"
                    },
                    {
                        "name": "Search Trend Analysis",
                        "description": "Changes in search volume for relevant topics",
                        "measurement_method": "Search engine trend data",
                        "change_significance": "Indicates emerging public interest",
                        "reliability": "Medium"
                    }
                ],
                "coincident_indicators": [
                    {
                        "name": "Public Opinion Polls",
                        "description": "Current public attitudes on relevant issues",
                        "measurement_method": "Representative polling",
                        "change_significance": "Reflects current public sentiment",
                        "reliability": "Medium"
                    },
                    {
                        "name": "Media Coverage Volume",
                        "description": "Amount of media attention to relevant topics",
                        "measurement_method": "Media monitoring services",
                        "change_significance": "Indicates current level of public attention",
                        "reliability": "Medium"
                    }
                ],
                "lagging_indicators": [
                    {
                        "name": "Behavioral Changes",
                        "description": "Actual changes in consumer or public behavior",
                        "measurement_method": "Consumer behavior studies",
                        "change_significance": "Confirms how attitudes translate to actions",
                        "reliability": "High"
                    },
                    {
                        "name": "Cultural References",
                        "description": "Incorporation of topics into cultural products",
                        "measurement_method": "Content analysis of media and entertainment",
                        "change_significance": "Confirms mainstream cultural integration",
                        "reliability": "Medium"
                    }
                ]
            }
        }
        
        # Default to General category if the specific category isn't in our fallbacks
        category = aspect_category if aspect_category in category_indicators else "Economic"
        
        return category_indicators[category]
    
    def _assess_indicator_quality(self, indicators):
        """
        Assess the quality of developed indicators.
        
        Args:
            indicators: Dictionary mapping aspect names to their indicators
            
        Returns:
            Dictionary containing indicator quality assessment
        """
        logger.info("Assessing indicator quality...")
        
        # Use data analysis MCP if available
        data_mcp = self.mcp_registry.get_mcp("data_analysis_mcp")
        
        if data_mcp:
            try:
                logger.info("Using data analysis MCP")
                indicator_assessment = data_mcp.assess_indicator_quality(indicators)
                return indicator_assessment
            except Exception as e:
                logger.error(f"Error using data analysis MCP: {e}")
                # Fall through to LLM-based assessment
        
        # Flatten all indicators into a single list
        all_indicators = []
        for aspect_name, aspect_indicators in indicators.items():
            for indicator_type in ["leading_indicators", "coincident_indicators", "lagging_indicators"]:
                for indicator in aspect_indicators.get(indicator_type, []):
                    all_indicators.append({
                        "aspect": aspect_name,
                        "type": indicator_type.replace("_indicators", ""),
                        "name": indicator.get("name", ""),
                        "description": indicator.get("description", ""),
                        "measurement_method": indicator.get("measurement_method", ""),
                        "reliability": indicator.get("reliability", "Medium")
                    })
        
        # Use LLM to assess indicator quality
        prompt = f"""
        Assess the quality of the following monitoring indicators:
        
        Indicators:
        {json.dumps(all_indicators, indent=2)}
        
        For this assessment:
        1. Identify the strongest indicators (most reliable and informative)
        2. Identify the weakest indicators (least reliable or informative)
        3. Identify any gaps in coverage (important aspects not well monitored)
        4. Suggest improvements for the indicator set as a whole
        
        Return your assessment as a JSON object with the following structure:
        {{
            "strongest_indicators": [
                {{
                    "indicator": "Name of the indicator",
                    "aspect": "Associated aspect",
                    "strengths": ["Strength 1", "Strength 2", ...]
                }},
                ...
            ],
            "weakest_indicators": [
                {{
                    "indicator": "Name of the indicator",
                    "aspect": "Associated aspect",
                    "weaknesses": ["Weakness 1", "Weakness 2", ...]
                }},
                ...
            ],
            "coverage_gaps": [
                {{
                    "description": "Description of the gap",
                    "suggested_indicators": ["Suggested indicator 1", "Suggested indicator 2", ...]
                }},
                ...
            ],
            "improvement_suggestions": ["Suggestion 1", "Suggestion 2", ...]
        }}
        """
        
        model_config = MODEL_CONFIG["sonar"]
        response = call_llm(prompt, model_config)
        content = extract_content(response)
        
        try:
            parsed_response = parse_json_response(content)
            
            if parsed_response.get("fallback_generated"):
                logger.error(f"Error assessing indicator quality: {parsed_response.get('error')}")
                return self._generate_fallback_indicator_assessment(all_indicators)
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing indicator quality assessment: {e}")
            return self._generate_fallback_indicator_assessment(all_indicators)
    
    def _generate_fallback_indicator_assessment(self, all_indicators):
        """
        Generate fallback indicator quality assessment when normal assessment fails.
        
        Args:
            all_indicators: List of all indicator dictionaries
            
        Returns:
            Dictionary containing fallback indicator quality assessment
        """
        # Select a few indicators to highlight as strongest/weakest
        strongest = all_indicators[:2] if len(all_indicators) >= 2 else all_indicators
        weakest = all_indicators[-2:] if len(all_indicators) >= 4 else all_indicators[-1:] if len(all_indicators) >= 3 else []
        
        return {
            "strongest_indicators": [
                {
                    "indicator": indicator.get("name", "Unknown"),
                    "aspect": indicator.get("aspect", "Unknown"),
                    "strengths": ["Directly measurable", "Reliable data source"]
                } for indicator in strongest
            ],
            "weakest_indicators": [
                {
                    "indicator": indicator.get("name", "Unknown"),
                    "aspect": indicator.get("aspect", "Unknown"),
                    "weaknesses": ["Potential measurement challenges", "Subjective interpretation"]
                } for indicator in weakest
            ],
            "coverage_gaps": [
                {
                    "description": "Insufficient coverage of international factors",
                    "suggested_indicators": ["International Policy Alignment Index", "Global Market Integration Measure"]
                },
                {
                    "description": "Limited long-term trend indicators",
                    "suggested_indicators": ["Structural Change Index", "Long-cycle Pattern Recognition"]
                }
            ],
            "improvement_suggestions": [
                "Develop more quantitative indicators with clear thresholds",
                "Include more indicators that can be automatically monitored",
                "Balance leading, coincident, and lagging indicators more evenly",
                "Add composite indicators that combine multiple data points"
            ]
        }
    
    def _develop_monitoring_plan(self, indicators, indicator_assessment):
        """
        Develop a plan for monitoring the indicators.
        
        Args:
            indicators: Dictionary mapping aspect names to their indicators
            indicator_assessment: Dictionary containing indicator quality assessment
            
        Returns:
            Dictionary containing monitoring plan
        """
        logger.info("Developing monitoring plan...")
        
        # Use monitoring MCP if available
        monitoring_mcp = self.mcp_registry.get_mcp("monitoring_mcp")
        
        if monitoring_mcp:
            try:
                logger.info("Using monitoring MCP")
                monitoring_plan = monitoring_mcp.develop_monitoring_plan(indicators, indicator_assessment)
                return monitoring_plan
            except Exception as e:
                logger.error(f"Error using monitoring MCP: {e}")
                # Fall through to LLM-based development
        
        # Flatten all indicators into a single list
        all_indicators = []
        for aspect_name, aspect_indicators in indicators.items():
            for indicator_type in ["leading_indicators", "coincident_indicators", "lagging_indicators"]:
                for indicator in aspect_indicators.get(indicator_type, []):
                    all_indicators.append({
                        "aspect": aspect_name,
                        "type": indicator_type.replace("_indicators", ""),
                        "name": indicator.get("name", ""),
                        "measurement_method": indicator.get("measurement_method", "")
                    })
        
        # Use LLM to develop monitoring plan
        prompt = f"""
        Develop a monitoring plan for the following indicators:
        
        Indicators:
        {json.dumps(all_indicators, indent=2)}
        
        Indicator Assessment:
        Strongest Indicators: {json.dumps([i.get("indicator", "") for i in indicator_assessment.get("strongest_indicators", [])], indent=2)}
        Weakest Indicators: {json.dumps([i.get("indicator", "") for i in indicator_assessment.get("weakest_indicators", [])], indent=2)}
        
        For this monitoring plan:
        1. Prioritize indicators for regular monitoring
        2. Specify appropriate monitoring frequency for different indicators
        3. Identify thresholds or patterns that would trigger alerts
        4. Suggest methods for efficient data collection and analysis
        5. Outline a process for reviewing and updating the indicator set
        
        Return your monitoring plan as a JSON object with the following structure:
        {{
            "priority_indicators": [
                {{
                    "indicator": "Name of the indicator",
                    "priority": "High/Medium/Low",
                    "rationale": "Explanation for this priority"
                }},
                ...
            ],
            "monitoring_schedule": [
                {{
                    "frequency": "Description of monitoring frequency (e.g., Daily, Weekly, Monthly)",
                    "indicators": ["Indicator 1", "Indicator 2", ...]
                }},
                ...
            ],
            "alert_thresholds": [
                {{
                    "indicator": "Name of the indicator",
                    "threshold": "Description of the threshold or pattern that would trigger an alert",
                    "response": "Recommended response to this alert"
                }},
                ...
            ],
            "data_collection_methods": ["Method 1", "Method 2", ...],
            "review_process": "Description of the process for reviewing and updating the indicator set"
        }}
        """
        
        model_config = MODEL_CONFIG["sonar"]
        response = call_llm(prompt, model_config)
        content = extract_content(response)
        
        try:
            parsed_response = parse_json_response(content)
            
            if parsed_response.get("fallback_generated"):
                logger.error(f"Error developing monitoring plan: {parsed_response.get('error')}")
                return self._generate_fallback_monitoring_plan(all_indicators)
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing monitoring plan: {e}")
            return self._generate_fallback_monitoring_plan(all_indicators)
    
    def _generate_fallback_monitoring_plan(self, all_indicators):
        """
        Generate fallback monitoring plan when normal development fails.
        
        Args:
            all_indicators: List of all indicator dictionaries
            
        Returns:
            Dictionary containing fallback monitoring plan
        """
        # Select a few indicators to highlight as priorities
        priority_indicators = all_indicators[:5] if len(all_indicators) >= 5 else all_indicators
        
        # Group indicators by type for monitoring frequency
        leading = [i.get("name", f"Leading Indicator {j+1}") for j, i in enumerate(all_indicators) if i.get("type", "") == "leading"]
        coincident = [i.get("name", f"Coincident Indicator {j+1}") for j, i in enumerate(all_indicators) if i.get("type", "") == "coincident"]
        lagging = [i.get("name", f"Lagging Indicator {j+1}") for j, i in enumerate(all_indicators) if i.get("type", "") == "lagging"]
        
        return {
            "priority_indicators": [
                {
                    "indicator": indicator.get("name", f"Priority Indicator {i+1}"),
                    "priority": "High" if i < 2 else "Medium",
                    "rationale": "Critical early warning indicator" if indicator.get("type", "") == "leading" else "Essential for current situation assessment" if indicator.get("type", "") == "coincident" else "Important for confirming trends"
                } for i, indicator in enumerate(priority_indicators)
            ],
            "monitoring_schedule": [
                {
                    "frequency": "Weekly",
                    "indicators": leading
                },
                {
                    "frequency": "Bi-weekly",
                    "indicators": coincident
                },
                {
                    "frequency": "Monthly",
                    "indicators": lagging
                }
            ],
            "alert_thresholds": [
                {
                    "indicator": priority_indicators[0].get("name", "Primary Indicator") if priority_indicators else "Key Indicator",
                    "threshold": "20% change from baseline within monitoring period",
                    "response": "Conduct detailed analysis and review related indicators"
                },
                {
                    "indicator": priority_indicators[1].get("name", "Secondary Indicator") if len(priority_indicators) > 1 else "Supporting Indicator",
                    "threshold": "Consistent directional change for three consecutive periods",
                    "response": "Update stakeholders and increase monitoring frequency"
                }
            ],
            "data_collection_methods": [
                "Automated data collection from public sources where possible",
                "Regular expert surveys for qualitative indicators",
                "Subscription to specialized data services for key metrics",
                "Periodic stakeholder interviews for context and interpretation"
            ],
            "review_process": "Quarterly review of indicator performance and relevance, with annual comprehensive reassessment of the entire indicator set. Ad hoc reviews triggered by significant environmental changes or when multiple alerts occur simultaneously."
        }
    
    def _generate_synthesis(self, question, indicators, indicator_assessment, monitoring_plan):
        """
        Generate a synthesis of the indicators development.
        
        Args:
            question: The analytical question
            indicators: Dictionary mapping aspect names to their indicators
            indicator_assessment: Dictionary containing indicator quality assessment
            monitoring_plan: Dictionary containing monitoring plan
            
        Returns:
            Dictionary containing the synthesis
        """
        logger.info("Generating synthesis of indicators development...")
        
        prompt = f"""
        Synthesize the following indicators development for the question:
        
        "{question}"
        
        Key Aspects Monitored:
        {json.dumps(list(indicators.keys()), indent=2)}
        
        Strongest Indicators:
        {json.dumps([i.get("indicator", "") for i in indicator_assessment.get("strongest_indicators", [])], indent=2)}
        
        Priority Indicators:
        {json.dumps([i.get("indicator", "") for i in monitoring_plan.get("priority_indicators", [])], indent=2)}
        
        Based on this indicators development:
        1. How effectively can the situation be monitored?
        2. What are the most important signals to watch?
        3. What blind spots or limitations remain?
        4. How should the indicators be used to inform decisions?
        
        Provide:
        1. A final judgment that addresses the original question
        2. A rationale for this judgment based on the indicators development
        3. A confidence level in this judgment (High/Medium/Low)
        4. Potential biases that might be affecting this analysis
        
        Return as JSON:
        {{
            "final_judgment": "Your assessment addressing the original question",
            "judgment_rationale": "Explanation for your judgment",
            "key_monitoring_recommendations": ["Recommendation 1", "Recommendation 2", ...],
            "critical_signals": ["Signal 1", "Signal 2", ...],
            "remaining_blind_spots": ["Blind spot 1", "Blind spot 2", ...],
            "confidence_level": "High/Medium/Low",
            "potential_biases": ["Bias 1", "Bias 2", ...]
        }}
        """
        
        model_config = MODEL_CONFIG["sonar_deep"]
        response = call_llm(prompt, model_config)
        content = extract_content(response)
        
        try:
            parsed_response = parse_json_response(content)
            
            if parsed_response.get("fallback_generated"):
                logger.error(f"Error generating synthesis: {parsed_response.get('error')}")
                return {
                    "final_judgment": "Error generating synthesis",
                    "judgment_rationale": parsed_response.get('error', "Unknown error"),
                    "key_monitoring_recommendations": ["Error in synthesis generation"],
                    "critical_signals": ["Error in synthesis generation"],
                    "remaining_blind_spots": ["Error in synthesis generation"],
                    "confidence_level": "Low",
                    "potential_biases": ["Technical error bias"]
                }
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing synthesis: {e}")
            return {
                "final_judgment": f"Error generating synthesis: {str(e)}",
                "judgment_rationale": "Error in synthesis generation",
                "key_monitoring_recommendations": ["Error in synthesis generation"],
                "critical_signals": ["Error in synthesis generation"],
                "remaining_blind_spots": ["Error in synthesis generation"],
                "confidence_level": "Low",
                "potential_biases": ["Technical error bias"]
            }
