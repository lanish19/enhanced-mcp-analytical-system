"""
Life Sciences & Biological Systems MCP for domain-specific analysis.
This module provides the LifeSciencesMCP class for specialized analysis in life sciences and biological systems.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional

from src.base_mcp import BaseMCP
from src.utils.llm_integration import call_llm, parse_json_response, LLMCallError, LLMParsingError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LifeSciencesMCP(BaseMCP):
    """
    Life Sciences & Biological Systems MCP for domain-specific analysis.
    
    This MCP provides specialized analysis capabilities for:
    1. Molecular biology and genetics
    2. Cellular and organismal biology
    3. Ecology and evolutionary biology
    4. Biodiversity and conservation
    5. Bioinformatics and systems biology
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the LifeSciencesMCP.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        super().__init__(
            name="LifeSciencesMCP",
            description="Specialized analysis in life sciences and biological systems",
            version="1.0.0"
        )
        
        self.config = config or {}
        self.api_key = self.config.get("api_key")
        self.model = self.config.get("model", "gpt-4o")
        
        # Operation handlers
        self.operation_handlers = {
            "analyze": self._analyze,
            "genetic_analysis": self._genetic_analysis,
            "ecological_assessment": self._ecological_assessment,
            "evolutionary_analysis": self._evolutionary_analysis,
            "biodiversity_assessment": self._biodiversity_assessment,
            "systems_biology_analysis": self._systems_biology_analysis
        }
        
        # Data sources (placeholders for now)
        self.data_sources = {
            "genetics": "NCBI GenBank API",
            "proteins": "UniProt API",
            "ecology": "GBIF Biodiversity API",
            "evolution": "TimeTree API",
            "systems_biology": "Reactome API"
        }
        
        logger.info("Initialized LifeSciencesMCP")
    
    def process(self, input_data: Dict) -> Dict:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Processing results
        """
        logger.info("Processing input in LifeSciencesMCP")
        
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
    
    def _analyze(self, input_data: Dict) -> Dict:
        """
        Perform general analysis in life sciences domain.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Analysis results
        """
        logger.info("Performing life sciences analysis")
        
        # Get question/text to analyze
        text = input_data.get("text", "")
        if not text:
            error_msg = "No text provided for analysis"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get context if provided
        context = input_data.get("context", {})
        
        # Get relevant research data if available
        research_data = input_data.get("research_data", "")
        
        # Construct prompt for domain-specific analysis
        system_prompt = """You are an expert in life sciences and biological systems with deep knowledge of 
        molecular biology, genetics, cellular biology, ecology, evolution, biodiversity, and systems biology. 
        Analyze the provided text from the perspective of life sciences, identifying relevant biological principles, 
        processes, and systems. Provide a structured analysis with biological context, key principles involved, 
        and implications based on established scientific understanding."""
        
        prompt = f"TEXT TO ANALYZE:\n{text}\n\n"
        
        if research_data:
            prompt += f"RELEVANT RESEARCH:\n{research_data}\n\n"
        
        prompt += """Please provide a comprehensive analysis from a life sciences perspective. 
        Structure your response as JSON with the following fields:
        - domain_assessment: Overall assessment from life sciences perspective
        - key_biological_principles: List of relevant biological principles and concepts
        - biological_processes: Key biological processes involved
        - ecological_factors: Relevant ecological dynamics (if applicable)
        - evolutionary_context: Evolutionary considerations (if applicable)
        - biological_uncertainties: Areas where biological understanding is incomplete
        - data_needs: Additional data that would improve the analysis
        - references: Key scientific references relevant to this analysis"""
        
        # Call LLM for analysis
        try:
            llm_response = call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.model,
                api_key=self.api_key,
                temperature=0.2
            )
            
            # Parse JSON response
            parsed_result = parse_json_response(llm_response)
            
            # Add metadata
            parsed_result["domain"] = "life_sciences"
            parsed_result["timestamp"] = time.time()
            parsed_result["data_sources"] = list(self.data_sources.values())
            
            return {
                "operation": "analyze",
                "input": text,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error in life sciences analysis: {str(e)}")
            return {
                "error": str(e),
                "operation": "analyze",
                "input": text
            }
    
    def _genetic_analysis(self, input_data: Dict) -> Dict:
        """
        Perform genetic analysis.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Genetic analysis results
        """
        logger.info("Performing genetic analysis")
        
        # Get genetic entity to analyze
        entity = input_data.get("entity", "")
        if not entity:
            error_msg = "No genetic entity provided for analysis"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # TODO: Implement data integration with genetic databases
        # For now, using placeholder for data integration
        genetic_data = f"PLACEHOLDER: Would integrate with NCBI GenBank API to retrieve genetic data for {entity}"
        
        # Construct prompt for genetic analysis
        system_prompt = """You are an expert geneticist and molecular biologist with deep knowledge of 
        genetics, genomics, gene expression, and molecular mechanisms. Analyze the specified genetic entity, 
        providing a comprehensive assessment based on current scientific understanding."""
        
        prompt = f"GENETIC ENTITY TO ANALYZE:\n{entity}\n\n"
        prompt += f"GENETIC DATA:\n{genetic_data}\n\n"
        prompt += """Please provide a comprehensive genetic analysis. Structure your response as JSON with the following fields:
        - genetic_assessment: Overall assessment from genetics perspective
        - molecular_structure: Key structural features and organization
        - functional_role: Known or predicted functional roles
        - expression_patterns: Expression patterns and regulation
        - evolutionary_conservation: Evolutionary conservation and variation
        - associated_phenotypes: Associated phenotypes or conditions
        - genetic_uncertainties: Areas of uncertainty in genetic understanding
        - references: Key genetic references relevant to this analysis"""
        
        # Call LLM for analysis
        try:
            llm_response = call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.model,
                api_key=self.api_key,
                temperature=0.2
            )
            
            # Parse JSON response
            parsed_result = parse_json_response(llm_response)
            
            # Add metadata
            parsed_result["domain"] = "genetics"
            parsed_result["timestamp"] = time.time()
            parsed_result["data_source"] = self.data_sources["genetics"]
            
            return {
                "operation": "genetic_analysis",
                "input": entity,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error in genetic analysis: {str(e)}")
            return {
                "error": str(e),
                "operation": "genetic_analysis",
                "input": entity
            }
    
    def _ecological_assessment(self, input_data: Dict) -> Dict:
        """
        Perform ecological assessment.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Ecological assessment results
        """
        logger.info("Performing ecological assessment")
        
        # Get ecosystem or ecological scenario to analyze
        ecosystem = input_data.get("ecosystem", "")
        if not ecosystem:
            error_msg = "No ecosystem specified for ecological assessment"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # TODO: Implement data integration with ecological databases
        # For now, using placeholder for data integration
        ecological_data = f"PLACEHOLDER: Would integrate with GBIF Biodiversity API for data on {ecosystem}"
        
        # Construct prompt for ecological assessment
        system_prompt = """You are an expert ecologist with deep knowledge of ecosystems, 
        ecological processes, community dynamics, and environmental interactions. Analyze the 
        specified ecosystem or ecological scenario, providing a comprehensive assessment based 
        on ecological principles and current scientific understanding."""
        
        prompt = f"ECOSYSTEM TO ANALYZE:\n{ecosystem}\n\n"
        prompt += f"ECOLOGICAL DATA:\n{ecological_data}\n\n"
        prompt += """Please provide a comprehensive ecological assessment. Structure your response as JSON with the following fields:
        - ecological_assessment: Overall assessment of the ecosystem
        - ecosystem_structure: Key structural components and organization
        - trophic_relationships: Food web and energy flow dynamics
        - species_interactions: Key species interactions and dependencies
        - ecosystem_services: Services provided by this ecosystem
        - disturbance_factors: Natural and anthropogenic disturbance factors
        - resilience_factors: Factors affecting ecosystem resilience
        - conservation_implications: Implications for conservation and management
        - ecological_uncertainties: Areas of uncertainty in ecological understanding
        - references: Key ecological references relevant to this analysis"""
        
        # Call LLM for analysis
        try:
            llm_response = call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.model,
                api_key=self.api_key,
                temperature=0.2
            )
            
            # Parse JSON response
            parsed_result = parse_json_response(llm_response)
            
            # Add metadata
            parsed_result["domain"] = "ecology"
            parsed_result["timestamp"] = time.time()
            parsed_result["data_source"] = self.data_sources["ecology"]
            
            return {
                "operation": "ecological_assessment",
                "input": ecosystem,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error in ecological assessment: {str(e)}")
            return {
                "error": str(e),
                "operation": "ecological_assessment",
                "input": ecosystem
            }
    
    def _evolutionary_analysis(self, input_data: Dict) -> Dict:
        """
        Perform evolutionary analysis.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Evolutionary analysis results
        """
        logger.info("Performing evolutionary analysis")
        
        # Get species or trait to analyze
        subject = input_data.get("subject", "")
        if not subject:
            error_msg = "No subject specified for evolutionary analysis"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # TODO: Implement data integration with evolutionary databases
        # For now, using placeholder for data integration
        evolutionary_data = f"PLACEHOLDER: Would integrate with TimeTree API for evolutionary data on {subject}"
        
        # Construct prompt for evolutionary analysis
        system_prompt = """You are an expert evolutionary biologist with deep knowledge of 
        evolutionary processes, phylogenetics, adaptation, and natural selection. Analyze the 
        specified species or trait from an evolutionary perspective, providing a comprehensive 
        assessment based on evolutionary principles and current scientific understanding."""
        
        prompt = f"SUBJECT FOR EVOLUTIONARY ANALYSIS:\n{subject}\n\n"
        prompt += f"EVOLUTIONARY DATA:\n{evolutionary_data}\n\n"
        prompt += """Please provide a comprehensive evolutionary analysis. Structure your response as JSON with the following fields:
        - evolutionary_assessment: Overall assessment from evolutionary perspective
        - phylogenetic_context: Evolutionary relationships and history
        - adaptive_significance: Adaptive value and selective pressures
        - evolutionary_mechanisms: Key evolutionary mechanisms involved
        - evolutionary_timeline: Approximate timeline of evolutionary events
        - convergent_evolution: Instances of convergent evolution (if applicable)
        - evolutionary_constraints: Constraints on evolutionary trajectories
        - evolutionary_uncertainties: Areas of uncertainty in evolutionary understanding
        - references: Key evolutionary biology references relevant to this analysis"""
        
        # Call LLM for analysis
        try:
            llm_response = call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.model,
                api_key=self.api_key,
                temperature=0.2
            )
            
            # Parse JSON response
            parsed_result = parse_json_response(llm_response)
            
            # Add metadata
            parsed_result["domain"] = "evolutionary_biology"
            parsed_result["timestamp"] = time.time()
            parsed_result["data_source"] = self.data_sources["evolution"]
            
            return {
                "operation": "evolutionary_analysis",
                "input": subject,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error in evolutionary analysis: {str(e)}")
            return {
                "error": str(e),
                "operation": "evolutionary_analysis",
                "input": subject
            }
    
    def _biodiversity_assessment(self, input_data: Dict) -> Dict:
        """
        Perform biodiversity assessment.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Biodiversity assessment results
        """
        logger.info("Performing biodiversity assessment")
        
        # Get region or habitat to analyze
        region = input_data.get("region", "")
        if not region:
            error_msg = "No region specified for biodiversity assessment"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # TODO: Implement data integration with biodiversity databases
        # For now, using placeholder for data integration
        biodiversity_data = f"PLACEHOLDER: Would integrate with GBIF Biodiversity API for data on {region}"
        
        # Construct prompt for biodiversity assessment
        system_prompt = """You are an expert conservation biologist with deep knowledge of 
        biodiversity patterns, conservation biology, biogeography, and ecosystem management. 
        Analyze the specified region or habitat from a biodiversity perspective, providing a 
        comprehensive assessment based on conservation principles and current scientific understanding."""
        
        prompt = f"REGION FOR BIODIVERSITY ASSESSMENT:\n{region}\n\n"
        prompt += f"BIODIVERSITY DATA:\n{biodiversity_data}\n\n"
        prompt += """Please provide a comprehensive biodiversity assessment. Structure your response as JSON with the following fields:
        - biodiversity_assessment: Overall assessment of biodiversity status
        - species_richness: Assessment of species richness and patterns
        - endemism: Endemic species and unique biodiversity elements
        - threatened_species: Threatened and endangered species
        - habitat_status: Status of key habitats and ecosystems
        - conservation_priorities: Key conservation priorities
        - threats_to_biodiversity: Major threats to biodiversity
        - conservation_strategies: Recommended conservation strategies
        - biodiversity_uncertainties: Areas of uncertainty in biodiversity understanding
        - references: Key conservation biology references relevant to this analysis"""
        
        # Call LLM for analysis
        try:
            llm_response = call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.model,
                api_key=self.api_key,
                temperature=0.2
            )
            
            # Parse JSON response
            parsed_result = parse_json_response(llm_response)
            
            # Add metadata
            parsed_result["domain"] = "conservation_biology"
            parsed_result["timestamp"] = time.time()
            parsed_result["data_source"] = self.data_sources["ecology"]
            
            return {
                "operation": "biodiversity_assessment",
                "input": region,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error in biodiversity assessment: {str(e)}")
            return {
                "error": str(e),
                "operation": "biodiversity_assessment",
                "input": region
            }
    
    def _systems_biology_analysis(self, input_data: Dict) -> Dict:
        """
        Perform systems biology analysis.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Systems biology analysis results
        """
        logger.info("Performing systems biology analysis")
        
        # Get biological system or pathway to analyze
        system = input_data.get("system", "")
        if not system:
            error_msg = "No biological system specified for systems biology analysis"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # TODO: Implement data integration with systems biology databases
        # For now, using placeholder for data integration
        systems_data = f"PLACEHOLDER: Would integrate with Reactome API for pathway data on {system}"
        
        # Construct prompt for systems biology analysis
        system_prompt = """You are an expert systems biologist with deep knowledge of 
        biological networks, pathways, computational biology, and integrative approaches. 
        Analyze the specified biological system or pathway, providing a comprehensive 
        assessment based on systems biology principles and current scientific understanding."""
        
        prompt = f"BIOLOGICAL SYSTEM TO ANALYZE:\n{system}\n\n"
        prompt += f"SYSTEMS BIOLOGY DATA:\n{systems_data}\n\n"
        prompt += """Please provide a comprehensive systems biology analysis. Structure your response as JSON with the following fields:
        - systems_assessment: Overall assessment from systems biology perspective
        - network_structure: Key structural features of the biological network
        - pathway_components: Major components and their interactions
        - regulatory_mechanisms: Key regulatory mechanisms and control points
        - emergent_properties: Emergent properties of the system
        - dynamic_behavior: Dynamic behavior and responses to perturbations
        - integration_with_other_systems: Integration with other biological systems
        - systems_uncertainties: Areas of uncertainty in systems understanding
        - computational_models: Relevant computational models and approaches
        - references: Key systems biology references relevant to this analysis"""
        
        # Call LLM for analysis
        try:
            llm_response = call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.model,
                api_key=self.api_key,
                temperature=0.2
            )
            
            # Parse JSON response
            parsed_result = parse_json_response(llm_response)
            
            # Add metadata
            parsed_result["domain"] = "systems_biology"
            parsed_result["timestamp"] = time.time()
            parsed_result["data_source"] = self.data_sources["systems_biology"]
            
            return {
                "operation": "systems_biology_analysis",
                "input": system,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error in systems biology analysis: {str(e)}")
            return {
                "error": str(e),
                "operation": "systems_biology_analysis",
                "input": system
            }
