"""
Premortem Analysis Technique implementation.
This module provides the PremortemAnalysisTechnique class for identifying potential failure modes.
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

class PremortemAnalysisTechnique(AnalyticalTechnique):
    """
    Identifies potential failure modes by imagining the analysis has failed.
    
    This technique works by assuming the analysis has already failed and then
    working backward to identify what could have caused the failure, helping
    to identify risks and preventive measures.
    """
    
    def execute(self, context, parameters):
        """
        Execute the premortem analysis technique.
        
        Args:
            context: The analysis context
            parameters: Parameters for the technique
            
        Returns:
            Dictionary containing premortem analysis results
        """
        logger.info(f"Executing PremortemAnalysisTechnique for question: {context.question[:50]}...")
        
        # Get parameters or use defaults
        target_analysis = parameters.get("target_analysis", "all")
        time_horizon = parameters.get("time_horizon", "6 months")
        
        # Step 1: Identify target content to analyze
        target_content = self._identify_target_content(context, target_analysis)
        
        # Step 2: Generate failure scenarios
        failure_scenarios = self._generate_failure_scenarios(context.question, target_content, time_horizon)
        
        # Step 3: Identify failure causes
        failure_causes = self._identify_failure_causes(failure_scenarios)
        
        # Step 4: Develop preventive measures
        preventive_measures = self._develop_preventive_measures(failure_causes)
        
        # Step 5: Generate final synthesis
        synthesis = self._generate_synthesis(context.question, failure_scenarios, failure_causes, preventive_measures)
        
        return {
            "technique": "Premortem Analysis",
            "status": "Completed",
            "failure_scenarios": failure_scenarios,
            "failure_causes": failure_causes,
            "preventive_measures": preventive_measures,
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
        return ["risk_analysis_mcp"]
    
    def _identify_target_content(self, context, target_analysis):
        """
        Identify the target content to analyze.
        
        Args:
            context: The analysis context
            target_analysis: Type of analysis to target (specific technique, all)
            
        Returns:
            Dictionary containing target content
        """
        logger.info(f"Identifying target content for analysis: {target_analysis}")
        
        if target_analysis in context.results:
            # Target a specific technique
            return {
                "type": "technique",
                "technique": target_analysis,
                "content": context.results[target_analysis],
                "question": context.question
            }
        else:
            # Target all available results
            all_content = {
                "type": "all",
                "question": context.question,
                "techniques": {}
            }
            
            for technique, result in context.results.items():
                all_content["techniques"][technique] = result
            
            return all_content
    
    def _generate_failure_scenarios(self, question, target_content, time_horizon):
        """
        Generate failure scenarios by imagining the analysis has failed.
        
        Args:
            question: The analytical question
            target_content: Target content to analyze
            time_horizon: Time horizon for the failure scenario
            
        Returns:
            List of failure scenario dictionaries
        """
        logger.info(f"Generating failure scenarios with time horizon: {time_horizon}...")
        
        # Use risk analysis MCP if available
        risk_mcp = self.mcp_registry.get_mcp("risk_analysis_mcp")
        
        if risk_mcp:
            try:
                logger.info("Using risk analysis MCP")
                failure_scenarios = risk_mcp.generate_failure_scenarios(question, target_content, time_horizon)
                return failure_scenarios
            except Exception as e:
                logger.error(f"Error using risk analysis MCP: {e}")
                # Fall through to LLM-based generation
        
        # Prepare content description for LLM-based generation
        content_description = ""
        
        if target_content["type"] == "technique":
            content_description = f"""
            Question: {target_content['question']}
            
            Technique: {target_content['technique']}
            
            Results: {json.dumps(target_content['content'], indent=2)}
            """
        else:  # target_content["type"] == "all"
            content_description = f"Question: {target_content['question']}\n\n"
            
            # Limit to a summary of each technique to avoid token limits
            for technique, result in target_content.get("techniques", {}).items():
                content_description += f"Technique: {technique}\n\n"
                
                # Extract key fields for summary
                summary = {}
                if "final_judgment" in result:
                    summary["final_judgment"] = result["final_judgment"]
                if "judgment_rationale" in result:
                    summary["judgment_rationale"] = result["judgment_rationale"]
                if "confidence_level" in result:
                    summary["confidence_level"] = result["confidence_level"]
                
                content_description += f"Results Summary: {json.dumps(summary, indent=2)}\n\n"
        
        # Use LLM to generate failure scenarios
        prompt = f"""
        Imagine it is {time_horizon} from now, and the following analysis has completely failed:
        
        {content_description}
        
        For this premortem analysis:
        1. Imagine the analysis has failed catastrophically
        2. Generate 4-5 distinct failure scenarios describing what went wrong
        3. For each scenario, describe the specific nature of the failure and its consequences
        4. Include a mix of analytical failures, implementation failures, and external context changes
        
        Return your response as a JSON object with the following structure:
        {{
            "failure_scenarios": [
                {{
                    "title": "Brief title for the failure scenario",
                    "description": "Detailed description of what went wrong",
                    "consequences": ["Consequence 1", "Consequence 2", ...],
                    "failure_type": "Analytical/Implementation/External Context"
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
                logger.error(f"Error generating failure scenarios: {parsed_response.get('error')}")
                return self._generate_fallback_failure_scenarios()
            
            failure_scenarios = parsed_response.get("failure_scenarios", [])
            
            if not failure_scenarios:
                logger.warning("No failure scenarios generated")
                return self._generate_fallback_failure_scenarios()
            
            return failure_scenarios
        
        except Exception as e:
            logger.error(f"Error parsing failure scenarios: {e}")
            return self._generate_fallback_failure_scenarios()
    
    def _generate_fallback_failure_scenarios(self):
        """
        Generate fallback failure scenarios when normal generation fails.
        
        Returns:
            List of fallback failure scenario dictionaries
        """
        return [
            {
                "title": "Incomplete Data Analysis",
                "description": "The analysis failed because critical data sources were overlooked or unavailable",
                "consequences": [
                    "Key insights were missed",
                    "Conclusions were based on partial information",
                    "Stakeholders lost confidence in the analysis"
                ],
                "failure_type": "Analytical"
            },
            {
                "title": "Flawed Assumptions",
                "description": "The analysis was built on assumptions that proved to be incorrect",
                "consequences": [
                    "Predictions failed to materialize",
                    "Resources were misallocated",
                    "Decision-making was compromised"
                ],
                "failure_type": "Analytical"
            },
            {
                "title": "Implementation Challenges",
                "description": "The recommendations from the analysis could not be effectively implemented",
                "consequences": [
                    "Practical application failed",
                    "Expected benefits were not realized",
                    "Stakeholders became frustrated with the process"
                ],
                "failure_type": "Implementation"
            },
            {
                "title": "External Disruption",
                "description": "Unexpected external events rendered the analysis obsolete",
                "consequences": [
                    "The context changed dramatically",
                    "Key assumptions became invalid",
                    "The analysis no longer applied to the new reality"
                ],
                "failure_type": "External Context"
            }
        ]
    
    def _identify_failure_causes(self, failure_scenarios):
        """
        Identify potential causes for each failure scenario.
        
        Args:
            failure_scenarios: List of failure scenario dictionaries
            
        Returns:
            List of failure cause dictionaries
        """
        logger.info(f"Identifying causes for {len(failure_scenarios)} failure scenarios...")
        
        failure_causes = []
        
        for scenario in failure_scenarios:
            logger.info(f"Identifying causes for scenario: {scenario.get('title', '')}...")
            
            prompt = f"""
            Identify potential causes for the following failure scenario:
            
            Title: {scenario.get('title', '')}
            Description: {scenario.get('description', '')}
            Consequences: {json.dumps(scenario.get('consequences', []), indent=2)}
            Failure Type: {scenario.get('failure_type', '')}
            
            For this failure scenario:
            1. Identify 3-5 distinct potential causes that could lead to this failure
            2. For each cause, assess its likelihood (High/Medium/Low)
            3. For each cause, identify early warning signs that might indicate this cause is developing
            
            Return your response as a JSON object with the following structure:
            {{
                "scenario_title": "Title of the failure scenario",
                "causes": [
                    {{
                        "description": "Description of the potential cause",
                        "likelihood": "High/Medium/Low",
                        "early_warning_signs": ["Warning sign 1", "Warning sign 2", ...]
                    }},
                    ...
                ]
            }}
            """
            
            model_config = MODEL_CONFIG["llama4"]
            response = call_llm(prompt, model_config)
            content = extract_content(response)
            
            try:
                parsed_response = parse_json_response(content)
                
                if parsed_response.get("fallback_generated"):
                    logger.error(f"Error identifying causes: {parsed_response.get('error')}")
                    causes = {
                        "scenario_title": scenario.get("title", "Unknown scenario"),
                        "causes": self._generate_fallback_causes()
                    }
                else:
                    causes = parsed_response
                
                failure_causes.append(causes)
            
            except Exception as e:
                logger.error(f"Error parsing causes: {e}")
                causes = {
                    "scenario_title": scenario.get("title", "Unknown scenario"),
                    "causes": self._generate_fallback_causes()
                }
                
                failure_causes.append(causes)
        
        return failure_causes
    
    def _generate_fallback_causes(self):
        """
        Generate fallback causes when normal identification fails.
        
        Returns:
            List of fallback cause dictionaries
        """
        return [
            {
                "description": "Insufficient data collection or analysis",
                "likelihood": "Medium",
                "early_warning_signs": [
                    "Data gaps identified during initial analysis",
                    "Difficulty answering basic questions about the subject",
                    "Overreliance on a limited number of sources"
                ]
            },
            {
                "description": "Cognitive biases affecting analysis",
                "likelihood": "Medium",
                "early_warning_signs": [
                    "Strong consensus without critical examination",
                    "Dismissal of contradictory evidence",
                    "Overconfidence in predictions"
                ]
            },
            {
                "description": "Changing external circumstances",
                "likelihood": "Medium",
                "early_warning_signs": [
                    "Early indicators of market or environmental shifts",
                    "Emerging trends that contradict assumptions",
                    "Unexpected stakeholder reactions"
                ]
            }
        ]
    
    def _develop_preventive_measures(self, failure_causes):
        """
        Develop preventive measures for identified failure causes.
        
        Args:
            failure_causes: List of failure cause dictionaries
            
        Returns:
            List of preventive measure dictionaries
        """
        logger.info(f"Developing preventive measures for {len(failure_causes)} failure scenarios...")
        
        preventive_measures = []
        
        for scenario_causes in failure_causes:
            scenario_title = scenario_causes.get("scenario_title", "Unknown scenario")
            causes = scenario_causes.get("causes", [])
            
            logger.info(f"Developing preventive measures for scenario: {scenario_title}...")
            
            # Prepare causes for the prompt
            causes_description = []
            for i, cause in enumerate(causes):
                causes_description.append(f"""
                Cause {i+1}: {cause.get('description', '')}
                Likelihood: {cause.get('likelihood', '')}
                Early Warning Signs: {json.dumps(cause.get('early_warning_signs', []), indent=2)}
                """)
            
            causes_text = "\n".join(causes_description)
            
            prompt = f"""
            Develop preventive measures for the following failure scenario and its causes:
            
            Scenario: {scenario_title}
            
            Causes:
            {causes_text}
            
            For this scenario:
            1. Develop 3-5 specific preventive measures that could mitigate these causes
            2. For each measure, assess its potential effectiveness (High/Medium/Low)
            3. For each measure, identify implementation challenges
            4. Suggest metrics or indicators to monitor the effectiveness of each measure
            
            Return your response as a JSON object with the following structure:
            {{
                "scenario_title": "Title of the failure scenario",
                "preventive_measures": [
                    {{
                        "description": "Description of the preventive measure",
                        "targeted_causes": ["Cause 1", "Cause 2", ...],
                        "effectiveness": "High/Medium/Low",
                        "implementation_challenges": ["Challenge 1", "Challenge 2", ...],
                        "monitoring_metrics": ["Metric 1", "Metric 2", ...]
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
                    logger.error(f"Error developing preventive measures: {parsed_response.get('error')}")
                    measures = {
                        "scenario_title": scenario_title,
                        "preventive_measures": self._generate_fallback_measures()
                    }
                else:
                    measures = parsed_response
                
                preventive_measures.append(measures)
            
            except Exception as e:
                logger.error(f"Error parsing preventive measures: {e}")
                measures = {
                    "scenario_title": scenario_title,
                    "preventive_measures": self._generate_fallback_measures()
                }
                
                preventive_measures.append(measures)
        
        return preventive_measures
    
    def _generate_fallback_measures(self):
        """
        Generate fallback preventive measures when normal development fails.
        
        Returns:
            List of fallback preventive measure dictionaries
        """
        return [
            {
                "description": "Implement robust data collection and validation processes",
                "targeted_causes": ["Insufficient data collection or analysis"],
                "effectiveness": "Medium",
                "implementation_challenges": [
                    "Resource constraints",
                    "Access to diverse data sources",
                    "Technical implementation complexity"
                ],
                "monitoring_metrics": [
                    "Data completeness score",
                    "Source diversity index",
                    "Validation success rate"
                ]
            },
            {
                "description": "Establish formal bias detection and mitigation procedures",
                "targeted_causes": ["Cognitive biases affecting analysis"],
                "effectiveness": "Medium",
                "implementation_challenges": [
                    "Resistance to acknowledging biases",
                    "Training requirements",
                    "Consistent application"
                ],
                "monitoring_metrics": [
                    "Bias detection rate",
                    "Alternative viewpoints considered",
                    "Confidence calibration scores"
                ]
            },
            {
                "description": "Implement continuous monitoring of key external factors",
                "targeted_causes": ["Changing external circumstances"],
                "effectiveness": "Medium",
                "implementation_challenges": [
                    "Identifying the right factors to monitor",
                    "Resource requirements for continuous monitoring",
                    "Timely integration of new information"
                ],
                "monitoring_metrics": [
                    "Early warning trigger rate",
                    "Response time to detected changes",
                    "Adaptation effectiveness score"
                ]
            }
        ]
    
    def _generate_synthesis(self, question, failure_scenarios, failure_causes, preventive_measures):
        """
        Generate a synthesis of the premortem analysis.
        
        Args:
            question: The analytical question
            failure_scenarios: List of failure scenario dictionaries
            failure_causes: List of failure cause dictionaries
            preventive_measures: List of preventive measure dictionaries
            
        Returns:
            Dictionary containing the synthesis
        """
        logger.info("Generating synthesis of premortem analysis...")
        
        # Prepare summaries for the prompt
        scenarios_summary = []
        for scenario in failure_scenarios:
            scenarios_summary.append(f"{scenario.get('title', '')}: {scenario.get('description', '')[:100]}...")
        
        causes_summary = []
        for scenario_causes in failure_causes:
            scenario_title = scenario_causes.get("scenario_title", "")
            causes = scenario_causes.get("causes", [])
            
            causes_list = []
            for cause in causes:
                causes_list.append(f"{cause.get('description', '')} (Likelihood: {cause.get('likelihood', '')})")
            
            causes_summary.append(f"{scenario_title}: {', '.join(causes_list)}")
        
        measures_summary = []
        for scenario_measures in preventive_measures:
            scenario_title = scenario_measures.get("scenario_title", "")
            measures = scenario_measures.get("preventive_measures", [])
            
            measures_list = []
            for measure in measures:
                measures_list.append(f"{measure.get('description', '')} (Effectiveness: {measure.get('effectiveness', '')})")
            
            measures_summary.append(f"{scenario_title}: {', '.join(measures_list)}")
        
        prompt = f"""
        Synthesize the following premortem analysis for the question:
        
        "{question}"
        
        Failure Scenarios:
        {json.dumps(scenarios_summary, indent=2)}
        
        Failure Causes:
        {json.dumps(causes_summary, indent=2)}
        
        Preventive Measures:
        {json.dumps(measures_summary, indent=2)}
        
        Based on this premortem analysis:
        1. What are the most critical risks to the analysis?
        2. What are the most effective preventive measures?
        3. How should the analysis be modified to increase its robustness?
        
        Provide:
        1. A final judgment about the robustness of the analysis and key risks
        2. A rationale for this judgment
        3. A confidence level in this judgment (High/Medium/Low)
        4. Potential biases that might be affecting this meta-analysis
        
        Return as JSON:
        {{
            "final_judgment": "Your assessment of the analysis robustness and key risks",
            "judgment_rationale": "Explanation for your judgment",
            "critical_risks": ["Risk 1", "Risk 2", ...],
            "key_preventive_measures": ["Measure 1", "Measure 2", ...],
            "recommended_modifications": ["Modification 1", "Modification 2", ...],
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
                    "critical_risks": ["Error in synthesis generation"],
                    "key_preventive_measures": ["Error in synthesis generation"],
                    "recommended_modifications": ["Error in synthesis generation"],
                    "confidence_level": "Low",
                    "potential_biases": ["Technical error bias"]
                }
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing synthesis: {e}")
            return {
                "final_judgment": f"Error generating synthesis: {str(e)}",
                "judgment_rationale": "Error in synthesis generation",
                "critical_risks": ["Error in synthesis generation"],
                "key_preventive_measures": ["Error in synthesis generation"],
                "recommended_modifications": ["Error in synthesis generation"],
                "confidence_level": "Low",
                "potential_biases": ["Technical error bias"]
            }
