"""
Delphistic Forecasting Technique implementation.
This module provides the DelphisticForecastingTechnique class for structured expert forecasting.
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

class DelphisticForecastingTechnique(AnalyticalTechnique):
    """
    Simulates a structured expert forecasting process.
    
    This technique emulates a Delphi method process where multiple expert
    perspectives are iteratively refined through structured feedback rounds
    to develop forecasts with quantified uncertainty.
    """
    
    def execute(self, context, parameters):
        """
        Execute the Delphistic forecasting technique.
        
        Args:
            context: The analysis context
            parameters: Parameters for the technique
            
        Returns:
            Dictionary containing Delphistic forecasting results
        """
        logger.info(f"Executing DelphisticForecastingTechnique for question: {context.question[:50]}...")
        
        # Get parameters or use defaults
        forecast_horizon = parameters.get("forecast_horizon", "1 year")
        num_experts = parameters.get("num_experts", 5)
        num_rounds = parameters.get("num_rounds", 3)
        expert_profiles = parameters.get("expert_profiles", None)
        
        # Step 1: Define forecast questions
        forecast_questions = self._define_forecast_questions(context, forecast_horizon)
        
        # Step 2: Generate expert profiles if not provided
        if not expert_profiles:
            expert_profiles = self._generate_expert_profiles(context.question, num_experts)
        
        # Step 3: Conduct initial round of forecasts
        initial_forecasts = self._conduct_forecast_round(context.question, forecast_questions, 
                                                        expert_profiles, round_num=1)
        
        # Step 4: Conduct subsequent rounds with feedback
        all_rounds = [initial_forecasts]
        current_forecasts = initial_forecasts
        
        for round_num in range(2, num_rounds + 1):
            current_forecasts = self._conduct_forecast_round(context.question, forecast_questions, 
                                                           expert_profiles, round_num=round_num, 
                                                           previous_forecasts=current_forecasts)
            all_rounds.append(current_forecasts)
        
        # Step 5: Analyze forecast convergence
        convergence_analysis = self._analyze_convergence(all_rounds)
        
        # Step 6: Generate final synthesis
        synthesis = self._generate_synthesis(context.question, forecast_questions, 
                                           all_rounds, convergence_analysis)
        
        return {
            "technique": "Delphistic Forecasting",
            "status": "Completed",
            "forecast_questions": forecast_questions,
            "expert_profiles": expert_profiles,
            "forecast_rounds": all_rounds,
            "convergence_analysis": convergence_analysis,
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
        return ["forecasting_mcp", "expert_simulation_mcp"]
    
    def _define_forecast_questions(self, context, forecast_horizon):
        """
        Define specific forecast questions based on the analytical question.
        
        Args:
            context: The analysis context
            forecast_horizon: Time horizon for forecasts
            
        Returns:
            List of forecast question dictionaries
        """
        logger.info(f"Defining forecast questions with horizon: {forecast_horizon}...")
        
        # Use forecasting MCP if available
        forecasting_mcp = self.mcp_registry.get_mcp("forecasting_mcp")
        
        if forecasting_mcp:
            try:
                logger.info("Using forecasting MCP")
                forecast_questions = forecasting_mcp.define_forecast_questions(context.question, forecast_horizon)
                return forecast_questions
            except Exception as e:
                logger.error(f"Error using forecasting MCP: {e}")
                # Fall through to LLM-based definition
        
        # Use LLM to define forecast questions
        prompt = f"""
        Define specific forecast questions for the following analytical question:
        
        "{context.question}"
        
        Forecast Horizon: {forecast_horizon}
        
        For this analysis:
        1. Identify 3-5 specific, measurable forecast questions that would help answer the analytical question
        2. Each forecast question should be:
           - Specific and unambiguous
           - Measurable or verifiable
           - Time-bound within the forecast horizon
           - Relevant to the analytical question
        3. Include a mix of different types of forecast questions (e.g., probability estimates, point estimates, ranges)
        
        Return your response as a JSON object with the following structure:
        {{
            "forecast_questions": [
                {{
                    "question": "Clear statement of the forecast question",
                    "type": "Type of forecast (Probability/Point Estimate/Range)",
                    "measurement": "How this forecast would be measured or verified",
                    "relevance": "How this forecast relates to the analytical question"
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
                logger.error(f"Error defining forecast questions: {parsed_response.get('error')}")
                return self._generate_fallback_forecast_questions(context.question, forecast_horizon)
            
            forecast_questions = parsed_response.get("forecast_questions", [])
            
            if not forecast_questions:
                logger.warning("No forecast questions defined")
                return self._generate_fallback_forecast_questions(context.question, forecast_horizon)
            
            return forecast_questions
        
        except Exception as e:
            logger.error(f"Error parsing forecast questions: {e}")
            return self._generate_fallback_forecast_questions(context.question, forecast_horizon)
    
    def _generate_fallback_forecast_questions(self, question, forecast_horizon):
        """
        Generate fallback forecast questions when definition fails.
        
        Args:
            question: The analytical question
            forecast_horizon: Time horizon for forecasts
            
        Returns:
            List of fallback forecast question dictionaries
        """
        return [
            {
                "question": f"What is the probability that the primary objective will be achieved within {forecast_horizon}?",
                "type": "Probability",
                "measurement": "Verification of objective completion by the end of the time period",
                "relevance": "Directly addresses the likelihood of success for the main goal"
            },
            {
                "question": f"What will be the rate of adoption/implementation by the end of {forecast_horizon}?",
                "type": "Point Estimate",
                "measurement": "Percentage of target population or systems that have adopted or implemented",
                "relevance": "Indicates the pace and extent of progress"
            },
            {
                "question": f"What range of resource requirements will be needed over the {forecast_horizon}?",
                "type": "Range",
                "measurement": "Minimum and maximum resource expenditure in relevant units",
                "relevance": "Helps plan for resource allocation and contingencies"
            },
            {
                "question": f"What is the probability of significant external disruption within {forecast_horizon}?",
                "type": "Probability",
                "measurement": "Occurrence of defined disruptive events during the time period",
                "relevance": "Addresses key risks and uncertainties"
            }
        ]
    
    def _generate_expert_profiles(self, question, num_experts):
        """
        Generate profiles for simulated experts.
        
        Args:
            question: The analytical question
            num_experts: Number of expert profiles to generate
            
        Returns:
            List of expert profile dictionaries
        """
        logger.info(f"Generating {num_experts} expert profiles...")
        
        # Use expert simulation MCP if available
        expert_mcp = self.mcp_registry.get_mcp("expert_simulation_mcp")
        
        if expert_mcp:
            try:
                logger.info("Using expert simulation MCP")
                expert_profiles = expert_mcp.generate_expert_profiles(question, num_experts)
                return expert_profiles
            except Exception as e:
                logger.error(f"Error using expert simulation MCP: {e}")
                # Fall through to LLM-based generation
        
        # Use LLM to generate expert profiles
        prompt = f"""
        Generate profiles for {num_experts} experts who would be qualified to forecast on the following question:
        
        "{question}"
        
        For each expert profile:
        1. Create a diverse set of backgrounds, expertise areas, and perspectives
        2. Include relevant qualifications and experience
        3. Note any potential biases or tendencies in their forecasting approach
        4. Assign a forecasting style (e.g., conservative, aggressive, data-driven, intuitive)
        
        Return your response as a JSON object with the following structure:
        {{
            "expert_profiles": [
                {{
                    "name": "Expert name",
                    "background": "Professional background and current role",
                    "expertise": ["Area of expertise 1", "Area of expertise 2", ...],
                    "perspective": "Brief description of their perspective or worldview",
                    "forecasting_style": "Description of their forecasting approach",
                    "potential_biases": ["Potential bias 1", "Potential bias 2", ...]
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
                logger.error(f"Error generating expert profiles: {parsed_response.get('error')}")
                return self._generate_fallback_expert_profiles(num_experts)
            
            expert_profiles = parsed_response.get("expert_profiles", [])
            
            if not expert_profiles or len(expert_profiles) < num_experts:
                logger.warning(f"Insufficient expert profiles generated: {len(expert_profiles)} < {num_experts}")
                return self._generate_fallback_expert_profiles(num_experts)
            
            return expert_profiles
        
        except Exception as e:
            logger.error(f"Error parsing expert profiles: {e}")
            return self._generate_fallback_expert_profiles(num_experts)
    
    def _generate_fallback_expert_profiles(self, num_experts):
        """
        Generate fallback expert profiles when generation fails.
        
        Args:
            num_experts: Number of expert profiles to generate
            
        Returns:
            List of fallback expert profile dictionaries
        """
        expert_types = [
            {
                "name": "Dr. Alex Morgan",
                "background": "Academic researcher with 15 years of experience in the field",
                "expertise": ["Statistical modeling", "Historical analysis", "Theoretical frameworks"],
                "perspective": "Cautious academic perspective with emphasis on methodological rigor",
                "forecasting_style": "Conservative, data-driven, emphasizes uncertainty",
                "potential_biases": ["Status quo bias", "Overemphasis on historical patterns"]
            },
            {
                "name": "Jordan Chen",
                "background": "Industry practitioner with 20 years of hands-on experience",
                "expertise": ["Practical implementation", "Operational constraints", "Market dynamics"],
                "perspective": "Pragmatic view focused on what works in practice",
                "forecasting_style": "Moderate, experience-based, focuses on practical outcomes",
                "potential_biases": ["Availability bias", "Overconfidence in personal experience"]
            },
            {
                "name": "Sam Rivera",
                "background": "Technology innovator and entrepreneur",
                "expertise": ["Emerging technologies", "Disruptive innovation", "Adoption patterns"],
                "perspective": "Forward-looking with emphasis on transformative change",
                "forecasting_style": "Aggressive, optimistic about change, focuses on possibilities",
                "potential_biases": ["Pro-innovation bias", "Underestimation of implementation challenges"]
            },
            {
                "name": "Dr. Taylor Kim",
                "background": "Policy analyst and government advisor",
                "expertise": ["Regulatory frameworks", "Public policy impacts", "Stakeholder analysis"],
                "perspective": "Institutional view with focus on governance and policy implications",
                "forecasting_style": "Methodical, context-sensitive, emphasizes structural factors",
                "potential_biases": ["Regulatory capture", "Institutional perspective bias"]
            },
            {
                "name": "Robin Patel",
                "background": "Independent consultant with cross-sector experience",
                "expertise": ["Comparative analysis", "Strategic planning", "Risk assessment"],
                "perspective": "Integrative view that synthesizes multiple perspectives",
                "forecasting_style": "Balanced, scenario-based, focuses on key uncertainties",
                "potential_biases": ["Recency bias", "Overemphasis on consensus"]
            }
        ]
        
        # Return the requested number of experts, cycling through the list if necessary
        return [expert_types[i % len(expert_types)] for i in range(num_experts)]
    
    def _conduct_forecast_round(self, question, forecast_questions, expert_profiles, round_num, previous_forecasts=None):
        """
        Conduct a round of forecasting with the simulated experts.
        
        Args:
            question: The analytical question
            forecast_questions: List of forecast question dictionaries
            expert_profiles: List of expert profile dictionaries
            round_num: Current round number
            previous_forecasts: Results from previous round (if applicable)
            
        Returns:
            Dictionary containing forecasts for this round
        """
        logger.info(f"Conducting forecast round {round_num}...")
        
        # Use expert simulation MCP if available
        expert_mcp = self.mcp_registry.get_mcp("expert_simulation_mcp")
        
        if expert_mcp:
            try:
                logger.info("Using expert simulation MCP")
                round_forecasts = expert_mcp.conduct_forecast_round(question, forecast_questions, 
                                                                  expert_profiles, round_num, 
                                                                  previous_forecasts)
                return round_forecasts
            except Exception as e:
                logger.error(f"Error using expert simulation MCP: {e}")
                # Fall through to LLM-based forecasting
        
        round_forecasts = {
            "round": round_num,
            "expert_forecasts": []
        }
        
        # For each expert, generate forecasts for all questions
        for expert in expert_profiles:
            expert_name = expert.get("name", "Unknown Expert")
            logger.info(f"Generating forecasts for expert: {expert_name}")
            
            # Prepare feedback from previous round if available
            feedback = ""
            if previous_forecasts and round_num > 1:
                feedback = self._prepare_feedback(previous_forecasts, expert_name)
            
            # Use LLM to generate expert forecasts
            prompt = f"""
            Generate forecasts for the following questions from the perspective of this expert:
            
            Analytical Question: "{question}"
            
            Expert Profile:
            {json.dumps(expert, indent=2)}
            
            Forecast Questions:
            {json.dumps(forecast_questions, indent=2)}
            
            {"" if round_num == 1 else f"Feedback from Round {round_num-1}:\n{feedback}"}
            
            This is Round {round_num} of a Delphi forecasting process.
            {"This is the initial round of forecasts." if round_num == 1 else f"In this round, the expert should consider the feedback from Round {round_num-1} and refine their forecasts accordingly."}
            
            For each forecast question:
            1. Provide the expert's forecast
            2. Include their confidence level (0-10)
            3. Provide their rationale for the forecast
            4. Note key factors they considered
            
            The forecast format should match the question type:
            - For probability questions: Percentage (0-100%)
            - For point estimates: Specific value with units
            - For ranges: Lower and upper bounds with units
            
            Return the expert's forecasts as a JSON object with the following structure:
            {{
                "expert_name": "Name of the expert",
                "forecasts": [
                    {{
                        "question": "The forecast question",
                        "forecast_value": "The expert's forecast",
                        "confidence": X,  # 0-10 scale
                        "rationale": "The expert's rationale for this forecast",
                        "key_factors": ["Factor 1", "Factor 2", ...]
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
                    logger.error(f"Error generating expert forecasts: {parsed_response.get('error')}")
                    expert_forecast = self._generate_fallback_expert_forecast(expert, forecast_questions, round_num)
                else:
                    expert_forecast = parsed_response
                
                round_forecasts["expert_forecasts"].append(expert_forecast)
            
            except Exception as e:
                logger.error(f"Error parsing expert forecasts: {e}")
                expert_forecast = self._generate_fallback_expert_forecast(expert, forecast_questions, round_num)
                round_forecasts["expert_forecasts"].append(expert_forecast)
        
        # Calculate round statistics
        round_forecasts["round_statistics"] = self._calculate_round_statistics(round_forecasts["expert_forecasts"], 
                                                                             forecast_questions)
        
        return round_forecasts
    
    def _prepare_feedback(self, previous_forecasts, expert_name):
        """
        Prepare feedback from previous round for a specific expert.
        
        Args:
            previous_forecasts: Results from previous round
            expert_name: Name of the expert to prepare feedback for
            
        Returns:
            String containing feedback for the expert
        """
        # Extract statistics from previous round
        statistics = previous_forecasts.get("round_statistics", {})
        
        # Find this expert's forecasts from previous round
        expert_forecasts = None
        for ef in previous_forecasts.get("expert_forecasts", []):
            if ef.get("expert_name") == expert_name:
                expert_forecasts = ef
                break
        
        if not expert_forecasts:
            return "No previous forecasts found for this expert."
        
        # Prepare feedback
        feedback_lines = ["Summary of previous round:"]
        
        for i, forecast in enumerate(expert_forecasts.get("forecasts", [])):
            question = forecast.get("question", f"Question {i+1}")
            expert_value = forecast.get("forecast_value", "N/A")
            
            # Get statistics for this question
            question_stats = {}
            for q_stat in statistics.get("question_statistics", []):
                if q_stat.get("question") == question:
                    question_stats = q_stat
                    break
            
            if question_stats:
                median = question_stats.get("median", "N/A")
                range_min = question_stats.get("range", {}).get("min", "N/A")
                range_max = question_stats.get("range", {}).get("max", "N/A")
                
                feedback_lines.append(f"Question: {question}")
                feedback_lines.append(f"Your forecast: {expert_value}")
                feedback_lines.append(f"Group median: {median}")
                feedback_lines.append(f"Group range: {range_min} to {range_max}")
                
                # Add outlier information if applicable
                if "outliers" in question_stats and expert_name in question_stats["outliers"]:
                    feedback_lines.append("Your forecast was identified as an outlier compared to the group.")
                
                feedback_lines.append("")
        
        return "\n".join(feedback_lines)
    
    def _generate_fallback_expert_forecast(self, expert, forecast_questions, round_num):
        """
        Generate fallback expert forecast when generation fails.
        
        Args:
            expert: Expert profile dictionary
            forecast_questions: List of forecast question dictionaries
            round_num: Current round number
            
        Returns:
            Dictionary containing fallback expert forecast
        """
        expert_name = expert.get("name", "Unknown Expert")
        forecasting_style = expert.get("forecasting_style", "").lower()
        
        # Adjust base values based on forecasting style
        confidence_modifier = 0
        value_modifier = 1.0
        if "conservative" in forecasting_style:
            confidence_modifier = -1
            value_modifier = 0.8
        elif "aggressive" in forecasting_style:
            confidence_modifier = 1
            value_modifier = 1.2
        
        # Adjust values based on round number (convergence in later rounds)
        if round_num > 1:
            confidence_modifier += 1
            value_modifier = 1.0 + (value_modifier - 1.0) * 0.5
        
        forecasts = []
        for i, question in enumerate(forecast_questions):
            question_text = question.get("question", f"Question {i+1}")
            question_type = question.get("type", "").lower()
            
            forecast_value = ""
            if "probability" in question_type:
                base_value = 50 + (i * 5)
                forecast_value = f"{int(base_value * value_modifier)}%"
            elif "point" in question_type:
                base_value = 100 + (i * 20)
                forecast_value = f"{int(base_value * value_modifier)} units"
            elif "range" in question_type:
                base_min = 80 + (i * 10)
                base_max = 120 + (i * 30)
                forecast_value = f"{int(base_min * value_modifier)} to {int(base_max * value_modifier)} units"
            else:
                forecast_value = "Unable to determine appropriate forecast format"
            
            confidence = 5 + confidence_modifier
            confidence = max(1, min(10, confidence))
            
            forecasts.append({
                "question": question_text,
                "forecast_value": forecast_value,
                "confidence": confidence,
                "rationale": f"Based on {expert.get('expertise', ['domain knowledge'])[0]} and consideration of key trends",
                "key_factors": [
                    "Historical patterns",
                    "Current trajectory",
                    "Potential disruptions",
                    f"Expert's {expert.get('perspective', 'perspective')}"
                ]
            })
        
        return {
            "expert_name": expert_name,
            "forecasts": forecasts
        }
    
    def _calculate_round_statistics(self, expert_forecasts, forecast_questions):
        """
        Calculate statistics for a round of forecasts.
        
        Args:
            expert_forecasts: List of expert forecast dictionaries
            forecast_questions: List of forecast question dictionaries
            
        Returns:
            Dictionary containing round statistics
        """
        logger.info("Calculating round statistics...")
        
        # Initialize statistics
        question_statistics = []
        
        # Process each forecast question
        for question_dict in forecast_questions:
            question = question_dict.get("question", "")
            question_type = question_dict.get("type", "").lower()
            
            # Collect all forecasts for this question
            all_values = []
            all_confidences = []
            expert_values = {}
            
            for expert_forecast in expert_forecasts:
                expert_name = expert_forecast.get("expert_name", "Unknown")
                
                for forecast in expert_forecast.get("forecasts", []):
                    if forecast.get("question") == question:
                        # Extract numeric value(s) from forecast
                        value = forecast.get("forecast_value", "")
                        confidence = forecast.get("confidence", 5)
                        
                        # Store raw value for expert
                        expert_values[expert_name] = value
                        
                        # Extract numeric values for statistics
                        numeric_values = self._extract_numeric_values(value, question_type)
                        if numeric_values:
                            # For probability and point estimates, use the single value
                            if "probability" in question_type or "point" in question_type:
                                all_values.append(numeric_values[0])
                            # For ranges, use the midpoint for central tendency
                            elif "range" in question_type and len(numeric_values) >= 2:
                                midpoint = (numeric_values[0] + numeric_values[1]) / 2
                                all_values.append(midpoint)
                        
                        all_confidences.append(confidence)
            
            # Calculate statistics if we have values
            if all_values:
                # Sort values for percentile calculations
                sorted_values = sorted(all_values)
                
                # Calculate basic statistics
                mean = sum(all_values) / len(all_values)
                median = sorted_values[len(sorted_values) // 2]
                if len(sorted_values) >= 2:
                    min_val = sorted_values[0]
                    max_val = sorted_values[-1]
                else:
                    min_val = mean * 0.9
                    max_val = mean * 1.1
                
                # Calculate standard deviation
                if len(all_values) >= 2:
                    variance = sum((x - mean) ** 2 for x in all_values) / len(all_values)
                    std_dev = variance ** 0.5
                else:
                    std_dev = 0
                
                # Identify outliers (values more than 1.5 std dev from mean)
                outliers = []
                for expert_name, value in expert_values.items():
                    numeric_values = self._extract_numeric_values(value, question_type)
                    if numeric_values:
                        # Use appropriate value for comparison
                        if "probability" in question_type or "point" in question_type:
                            compare_value = numeric_values[0]
                        elif "range" in question_type and len(numeric_values) >= 2:
                            compare_value = (numeric_values[0] + numeric_values[1]) / 2
                        else:
                            continue
                        
                        if abs(compare_value - mean) > 1.5 * std_dev:
                            outliers.append(expert_name)
                
                # Calculate average confidence
                avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 5
                
                # Store statistics for this question
                question_statistics.append({
                    "question": question,
                    "mean": mean,
                    "median": median,
                    "range": {
                        "min": min_val,
                        "max": max_val
                    },
                    "standard_deviation": std_dev,
                    "outliers": outliers,
                    "average_confidence": avg_confidence,
                    "num_forecasts": len(all_values)
                })
            else:
                # No valid values found
                question_statistics.append({
                    "question": question,
                    "error": "No valid numeric values found in forecasts",
                    "num_forecasts": 0
                })
        
        return {
            "question_statistics": question_statistics,
            "num_experts": len(expert_forecasts)
        }
    
    def _extract_numeric_values(self, value_str, question_type):
        """
        Extract numeric values from a forecast value string.
        
        Args:
            value_str: String containing the forecast value
            question_type: Type of forecast question
            
        Returns:
            List of numeric values extracted from the string
        """
        if not value_str:
            return []
        
        # For probability questions, extract percentage
        if "probability" in question_type.lower():
            # Extract numbers and remove % signs
            value_str = value_str.replace("%", "")
            try:
                return [float(value_str)]
            except ValueError:
                # Try to find any number in the string
                import re
                numbers = re.findall(r'\d+\.?\d*', value_str)
                if numbers:
                    return [float(numbers[0])]
                return []
        
        # For point estimates and ranges, extract numbers with units
        import re
        numbers = re.findall(r'\d+\.?\d*', value_str)
        return [float(n) for n in numbers]
    
    def _analyze_convergence(self, all_rounds):
        """
        Analyze the convergence of forecasts across rounds.
        
        Args:
            all_rounds: List of dictionaries containing results from each round
            
        Returns:
            Dictionary containing convergence analysis
        """
        logger.info("Analyzing forecast convergence...")
        
        # Use forecasting MCP if available
        forecasting_mcp = self.mcp_registry.get_mcp("forecasting_mcp")
        
        if forecasting_mcp:
            try:
                logger.info("Using forecasting MCP")
                convergence_analysis = forecasting_mcp.analyze_convergence(all_rounds)
                return convergence_analysis
            except Exception as e:
                logger.error(f"Error using forecasting MCP: {e}")
                # Fall through to LLM-based analysis
        
        # Extract questions from the first round
        if not all_rounds:
            return {"error": "No forecast rounds available for analysis"}
        
        first_round = all_rounds[0]
        if "round_statistics" not in first_round or "question_statistics" not in first_round["round_statistics"]:
            return {"error": "Invalid round statistics format"}
        
        questions = [qs.get("question") for qs in first_round["round_statistics"]["question_statistics"]]
        
        # Prepare data for convergence analysis
        question_data = {}
        for question in questions:
            question_data[question] = {
                "means": [],
                "medians": [],
                "std_devs": [],
                "ranges": [],
                "confidences": []
            }
        
        # Collect statistics across rounds
        for round_data in all_rounds:
            round_stats = round_data.get("round_statistics", {})
            question_stats = round_stats.get("question_statistics", [])
            
            for qs in question_stats:
                question = qs.get("question")
                if question in question_data:
                    question_data[question]["means"].append(qs.get("mean"))
                    question_data[question]["medians"].append(qs.get("median"))
                    question_data[question]["std_devs"].append(qs.get("standard_deviation"))
                    
                    range_min = qs.get("range", {}).get("min")
                    range_max = qs.get("range", {}).get("max")
                    if range_min is not None and range_max is not None:
                        question_data[question]["ranges"].append(range_max - range_min)
                    
                    question_data[question]["confidences"].append(qs.get("average_confidence"))
        
        # Analyze convergence for each question
        convergence_results = []
        
        for question, data in question_data.items():
            # Check if we have enough rounds for analysis
            if len(data["means"]) < 2:
                convergence_results.append({
                    "question": question,
                    "convergence_level": "Unknown",
                    "explanation": "Insufficient rounds for convergence analysis"
                })
                continue
            
            # Calculate convergence metrics
            std_dev_change = data["std_devs"][-1] - data["std_devs"][0] if data["std_devs"][0] else 0
            range_change = data["ranges"][-1] - data["ranges"][0] if data["ranges"] and data["ranges"][0] else 0
            confidence_change = data["confidences"][-1] - data["confidences"][0] if data["confidences"] else 0
            
            # Determine convergence level
            if std_dev_change < 0 and range_change < 0 and confidence_change > 0:
                convergence_level = "High"
                explanation = "Strong convergence with decreasing variance and increasing confidence"
            elif std_dev_change < 0 or range_change < 0:
                convergence_level = "Moderate"
                explanation = "Partial convergence with some decreasing variance"
            elif confidence_change > 0:
                convergence_level = "Low"
                explanation = "Limited convergence with increasing confidence but stable variance"
            else:
                convergence_level = "None"
                explanation = "No evidence of convergence across rounds"
            
            # Store results
            convergence_results.append({
                "question": question,
                "convergence_level": convergence_level,
                "explanation": explanation,
                "metrics": {
                    "final_mean": data["means"][-1],
                    "final_median": data["medians"][-1],
                    "std_dev_change": std_dev_change,
                    "range_change": range_change,
                    "confidence_change": confidence_change
                }
            })
        
        # Calculate overall convergence
        convergence_levels = [r.get("convergence_level") for r in convergence_results]
        high_count = convergence_levels.count("High")
        moderate_count = convergence_levels.count("Moderate")
        low_count = convergence_levels.count("Low")
        none_count = convergence_levels.count("None")
        
        if high_count > len(convergence_levels) / 2:
            overall_convergence = "High"
        elif high_count + moderate_count > len(convergence_levels) / 2:
            overall_convergence = "Moderate to High"
        elif moderate_count > len(convergence_levels) / 2:
            overall_convergence = "Moderate"
        elif moderate_count + low_count > len(convergence_levels) / 2:
            overall_convergence = "Low to Moderate"
        else:
            overall_convergence = "Low"
        
        return {
            "question_convergence": convergence_results,
            "overall_convergence": overall_convergence,
            "num_rounds": len(all_rounds)
        }
    
    def _generate_synthesis(self, question, forecast_questions, all_rounds, convergence_analysis):
        """
        Generate a synthesis of the Delphistic forecasting.
        
        Args:
            question: The analytical question
            forecast_questions: List of forecast question dictionaries
            all_rounds: List of dictionaries containing results from each round
            convergence_analysis: Dictionary containing convergence analysis
            
        Returns:
            Dictionary containing the synthesis
        """
        logger.info("Generating synthesis of Delphistic forecasting...")
        
        # Extract final round results
        final_round = all_rounds[-1] if all_rounds else {}
        final_stats = final_round.get("round_statistics", {}).get("question_statistics", [])
        
        # Extract convergence information
        question_convergence = convergence_analysis.get("question_convergence", [])
        overall_convergence = convergence_analysis.get("overall_convergence", "Unknown")
        
        # Prepare summary of final forecasts
        final_forecasts = []
        for qs in final_stats:
            question_text = qs.get("question", "")
            
            # Find corresponding convergence info
            convergence_info = {}
            for qc in question_convergence:
                if qc.get("question") == question_text:
                    convergence_info = qc
                    break
            
            final_forecasts.append({
                "question": question_text,
                "median": qs.get("median"),
                "range": qs.get("range", {}),
                "confidence": qs.get("average_confidence"),
                "convergence": convergence_info.get("convergence_level", "Unknown")
            })
        
        # Use LLM to generate synthesis
        prompt = f"""
        Synthesize the following Delphistic forecasting results for the question:
        
        "{question}"
        
        Final Forecasts:
        {json.dumps(final_forecasts, indent=2)}
        
        Overall Convergence: {overall_convergence}
        
        Based on this Delphistic forecasting:
        1. What are the key findings from the expert forecasts?
        2. How reliable are these forecasts based on convergence and confidence?
        3. What implications do these forecasts have for the original question?
        4. What uncertainties or limitations should be noted?
        
        Provide:
        1. A final judgment that addresses the original question
        2. A rationale for this judgment based on the Delphistic forecasting
        3. A confidence level in this judgment (High/Medium/Low)
        4. Potential biases that might be affecting this analysis
        
        Return as JSON:
        {{
            "final_judgment": "Your assessment addressing the original question",
            "judgment_rationale": "Explanation for your judgment",
            "key_findings": ["Finding 1", "Finding 2", ...],
            "reliability_assessment": "Assessment of the reliability of the forecasts",
            "implications": ["Implication 1", "Implication 2", ...],
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
                    "key_findings": ["Error in synthesis generation"],
                    "reliability_assessment": "Unable to assess reliability due to synthesis error",
                    "implications": ["Error in synthesis generation"],
                    "confidence_level": "Low",
                    "potential_biases": ["Technical error bias"]
                }
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing synthesis: {e}")
            return {
                "final_judgment": f"Error generating synthesis: {str(e)}",
                "judgment_rationale": "Error in synthesis generation",
                "key_findings": ["Error in synthesis generation"],
                "reliability_assessment": "Unable to assess reliability due to synthesis error",
                "implications": ["Error in synthesis generation"],
                "confidence_level": "Low",
                "potential_biases": ["Technical error bias"]
            }
