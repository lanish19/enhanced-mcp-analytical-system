"""
Causal Network Analysis Technique implementation.
This module provides the CausalNetworkAnalysisTechnique class for mapping causal relationships.
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

class CausalNetworkAnalysisTechnique(AnalyticalTechnique):
    """
    Maps causal relationships between factors to identify drivers, effects, and feedback loops.
    
    This technique identifies key entities and factors related to the question,
    maps the causal relationships between them, and analyzes the resulting network
    to identify central nodes, feedback loops, and potential intervention points.
    """
    
    def execute(self, context, parameters):
        """
        Execute the causal network analysis technique.
        
        Args:
            context: The analysis context
            parameters: Parameters for the technique
            
        Returns:
            Dictionary containing causal network analysis results
        """
        logger.info(f"Executing CausalNetworkAnalysisTechnique for question: {context.question[:50]}...")
        
        # Get parameters or use defaults
        #max_entities = parameters.get("max_entities", 10)
        include_feedback_loops = parameters.get("include_feedback_loops", True)
        
        # Step 1: Extract entities and concepts
        entities = self._extract_entities(context)
        
        # Step 2: Identify causal relationships
        relationships = self._identify_relationships(context.question, entities)

        # Fetch Economic Data and Geopolitical Data
        economic_data = self._fetch_economic_data(context.question)
        geopolitical_data = self._fetch_geopolitical_data(context.question)

        relationships = self._identify_relationships(context.question, entities, economic_data, geopolitical_data)
        # Step 3: Build and analyze causal network
        network_analysis = self._analyze_network(relationships, include_feedback_loops)
        
        # Step 4: Identify potential intervention points
        interventions = self._identify_interventions(network_analysis)
        
        # Step 5: Generate final synthesis
        synthesis = self._generate_synthesis(context.question, entities, relationships, network_analysis, interventions, economic_data, geopolitical_data)
        
        return {
            "technique": "Causal Network Analysis",
            "status": "Completed",
            "entities": entities,
            "relationships": relationships,
            "network_analysis": network_analysis,
            "potential_interventions": interventions,
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
        return ["entity_extraction_mcp", "relationship_analysis_mcp"]
    
    def _extract_entities(self, context):
        """
        Extract entities and concepts from the question and context.
        
        Args:
            context: The analysis context
            
        Returns:
            List of entity dictionaries
        """
        logger.info("Extracting entities and concepts...")
        
        # Use entity extraction MCP if available
        entity_mcp = self.mcp_registry.get_mcp("entity_extraction_mcp")
        
        if entity_mcp:
            try:
                logger.info("Using entity extraction MCP")
                entities = entity_mcp.extract_entities(context.question)
                return entities
            except Exception as e:
                logger.error(f"Error using entity extraction MCP: {e}")
                # Fall through to LLM-based extraction
        
        # Extract additional entities from research results if available
        additional_entities = []
        if "research_to_hypothesis" in context.results:
            research_data = context.results["research_to_hypothesis"]
            try:
                additional_entities = self._extract_entities_from_research(research_data)
            except Exception as e:
                logger.error(f"Error extracting entities from research: {e}")
        
        # Use LLM to extract entities
        prompt = f"""
        Extract the key entities, factors, and concepts from the following analytical question:
        
        "{context.question}"
        
        For each entity/factor:
        1. Provide a clear name
        2. Categorize it (e.g., economic factor, political actor, technological trend, etc.)
        3. Provide a brief description
        
        Return your response as a JSON object with the following structure:
        {{
            "entities": [
                {{
                    "name": "Entity name",
                    "category": "Entity category",
                    "description": "Brief description"
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
                logger.error(f"Error extracting entities: {parsed_response.get('error')}")
                return self._generate_fallback_entities(context.question)
            
            entities = parsed_response.get("entities", [])
            
            # Add additional entities from research
            for entity in additional_entities:
                if entity not in entities:
                    entities.append(entity)
            
            return entities
        
        except Exception as e:
            logger.error(f"Error parsing entities: {e}")
            return self._generate_fallback_entities(context.question)
    
    def _extract_entities_from_research(self, research_data):
        """
        Extract entities from research results.
        
        Args:
            research_data: Research results data
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        
        # Extract from key findings
        key_findings = research_data.get("key_findings", [])
        if key_findings:
            prompt = f"""
            Extract key entities and factors from the following research findings:
            
            {json.dumps(key_findings, indent=2)}
            
            For each entity/factor:
            1. Provide a clear name
            2. Categorize it (e.g., economic factor, political actor, technological trend, etc.)
            3. Provide a brief description
            
            Return your response as a JSON object with the following structure:
            {{
                "entities": [
                    {{
                        "name": "Entity name",
                        "category": "Entity category",
                        "description": "Brief description"
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
                
                if not parsed_response.get("fallback_generated"):
                    entities.extend(parsed_response.get("entities", []))
            
            except Exception as e:
                logger.error(f"Error parsing entities from research: {e}")
        
        return entities
    
    def _generate_fallback_entities(self, question):
        """
        Generate fallback entities when extraction fails.
        
        Args:
            question: The analytical question
            
        Returns:
            List of fallback entity dictionaries
        """
        return [
            {
                "name": "Primary Factor",
                "category": "Main concept",
                "description": f"The primary subject of the question: '{question[:50]}...'"
            },
            {
                "name": "Secondary Factor",
                "category": "Related concept",
                "description": "A secondary factor related to the primary subject"
            },
            {
                "name": "External Influence",
                "category": "External factor",
                "description": "An external factor that may influence the situation"
            }
        ]
    
    def _identify_relationships(self, question, entities, economic_data=None, geopolitical_data=None):
        """
        Identify causal relationships between entities.
        
        Args:
            question: The analytical question
            entities: List of entity dictionaries
            
        Returns:
            List of relationship dictionaries
        """
        logger.info(f"Identifying causal relationships between {len(entities)} entities...")
        
        # Use relationship analysis MCP if available
        relationship_mcp = self.mcp_registry.get_mcp("relationship_analysis_mcp")
        
        if relationship_mcp:
            try:
                logger.info("Using relationship analysis MCP")
                relationships = relationship_mcp.identify_causal_relationships(entities, question)
                return relationships
            except Exception as e:
                logger.error(f"Error using relationship analysis MCP: {e}")
                # Fall through to LLM-based relationship identification
        
        # Use LLM to identify relationships
        entity_names = [entity.get("name", f"Entity {i+1}") for i, entity in enumerate(entities)]
        
        prompt = f"""
        Identify the causal relationships between the following entities related to the question: 
        
        Question: "{question}"
        
        Entities:
        {json.dumps(entities, indent=2)}

        """
        if economic_data:
            prompt += f"""
            Relevant Economic Data:
            {json.dumps(economic_data, indent=2)}
            """

        if geopolitical_data:
            prompt += f"""
            Relevant Geopolitical Data:
            {json.dumps(geopolitical_data, indent=2)}
            """
        
        For each relationship:
        1. Identify the source entity (cause)
        2. Identify the target entity (effect)
        3. Describe the nature of the causal relationship
        4. Assess the strength of the relationship (Strong/Moderate/Weak)
        5. Indicate the direction (Positive/Negative/Mixed)
        
        Return your response as a JSON object with the following structure:
        {{
            "relationships": [
                {{
                    "source": "Source entity name",
                    "target": "Target entity name",
                    "description": "Description of the causal relationship",
                    "strength": "Strong/Moderate/Weak",
                    "direction": "Positive/Negative/Mixed"
                }},
                ...
            ]
        }}
        
        Note: Not all entities need to be connected, and some entities may have multiple relationships.
        """
        
        model_config = MODEL_CONFIG["sonar"]
        response = call_llm(prompt, model_config)
        content = extract_content(response)
        
        try:
            parsed_response = parse_json_response(content)
            
            if parsed_response.get("fallback_generated"):
                logger.error(f"Error identifying relationships: {parsed_response.get('error')}")
                return self._generate_fallback_relationships(entities)
            
            relationships = parsed_response.get("relationships", [])
            
            # Validate relationships
            valid_relationships = []
            entity_names_set = set(entity_names)
            
            for rel in relationships:
                source = rel.get("source")
                target = rel.get("target")
                
                if source in entity_names_set and target in entity_names_set and source != target:
                    valid_relationships.append(rel)
                else:
                    logger.warning(f"Invalid relationship: {source} -> {target}")
            
            if not valid_relationships:
                logger.warning("No valid relationships identified")
                return self._generate_fallback_relationships(entities)
            
            return valid_relationships
        
        except Exception as e:
            logger.error(f"Error parsing relationships: {e}")
            return self._generate_fallback_relationships(entities)
    
    def _generate_fallback_relationships(self, entities):
        """
        Generate fallback relationships when identification fails.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            List of fallback relationship dictionaries
        """
        fallback_relationships = []
        
        if len(entities) < 2:
            return fallback_relationships
        
        # Create some basic relationships between entities
        for i in range(len(entities) - 1):
            source = entities[i].get("name", f"Entity {i+1}")
            target = entities[i+1].get("name", f"Entity {i+2}")
            
            fallback_relationships.append({
                "source": source,
                "target": target,
                "description": f"Potential causal relationship from {source} to {target}",
                "strength": "Moderate",
                "direction": "Positive"
            })
        
        # Add a relationship from the last to the first entity if there are at least 3 entities
        if len(entities) >= 3:
            source = entities[-1].get("name", f"Entity {len(entities)}")
            target = entities[0].get("name", "Entity 1")
            
            fallback_relationships.append({
                "source": source,
                "target": target,
                "description": f"Potential causal relationship from {source} to {target}",
                "strength": "Weak",
                "direction": "Negative"
            })
        
        return fallback_relationships
    
    def _analyze_network(self, relationships, include_feedback_loops):
        """
        Analyze the causal network to identify key properties.
        
        Args:
            relationships: List of relationship dictionaries
            include_feedback_loops: Whether to identify feedback loops
            
        Returns:
            Dictionary containing network analysis results
        """
        logger.info("Analyzing causal network...")
        
        # Build adjacency list representation of the network
        network = {}
        all_nodes = set()
        
        for rel in relationships:
            source = rel.get("source")
            target = rel.get("target")
            
            if source and target:
                all_nodes.add(source)
                all_nodes.add(target)
                
                if source not in network:
                    network[source] = []
                
                network[source].append({
                    "target": target,
                    "description": rel.get("description", ""),
                    "strength": rel.get("strength", "Moderate"),
                    "direction": rel.get("direction", "Positive")
                })
        
        # Add nodes with no outgoing edges
        for node in all_nodes:
            if node not in network:
                network[node] = []
        
        # Identify central nodes (high out-degree)
        central_nodes = []
        for node, edges in network.items():
            if len(edges) >= 2:  # Nodes with at least 2 outgoing edges
                central_nodes.append({
                    "name": node,
                    "out_degree": len(edges),
                    "targets": [edge.get("target") for edge in edges]
                })
        
        # Sort central nodes by out-degree
        central_nodes.sort(key=lambda x: x.get("out_degree", 0), reverse=True)
        
        # Identify end nodes (no outgoing edges)
        end_nodes = [node for node, edges in network.items() if not edges]
        
        # Identify feedback loops if requested
        feedback_loops = []
        if include_feedback_loops:
            feedback_loops = self._identify_feedback_loops(network)
        
        return {
            "nodes": list(all_nodes),
            "central_nodes": central_nodes,
            "end_nodes": end_nodes,
            "feedback_loops": feedback_loops
        }
    
    def _identify_feedback_loops(self, network):
        """
        Identify feedback loops in the causal network.
        
        Args:
            network: Adjacency list representation of the network
            
        Returns:
            List of feedback loop dictionaries
        """
        logger.info("Identifying feedback loops...")
        
        # Simple DFS-based cycle detection
        feedback_loops = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                
                # Check if this cycle is already found (might be the same cycle from different starting points)
                cycle_signature = '->'.join(sorted(cycle))
                if not any(loop.get("signature") == cycle_signature for loop in feedback_loops):
                    feedback_loops.append({
                        "nodes": cycle,
                        "description": f"Feedback loop involving {', '.join(cycle)}",
                        "signature": cycle_signature
                    })
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for edge in network.get(node, []):
                target = edge.get("target")
                if target:
                    dfs(target, path.copy())
            
            rec_stack.remove(node)
        
        # Run DFS from each node
        for node in network:
            dfs(node, [])
        
        # Remove the signature field used for deduplication
        for loop in feedback_loops:
            if "signature" in loop:
                del loop["signature"]
        
        return feedback_loops
    
    def _identify_interventions(self, network_analysis):
        """
        Identify potential intervention points in the causal network.
        
        Args:
            network_analysis: Dictionary containing network analysis results
            
        Returns:
            List of intervention dictionaries
        """
        logger.info("Identifying potential intervention points...")
        
        interventions = []
        
        # Central nodes are often good intervention points
        for node in network_analysis.get("central_nodes", []):
            interventions.append({
                "node": node.get("name"),
                "type": "Central node",
                "rationale": f"High influence node with {node.get('out_degree', 0)} outgoing connections",
                "potential_impact": "High"
            })
        
        # Nodes in feedback loops can be intervention points to break reinforcing cycles
        for loop in network_analysis.get("feedback_loops", []):
            loop_nodes = loop.get("nodes", [])
            if loop_nodes:
                interventions.append({
                    "node": loop_nodes[0],  # Just pick the first node in the loop
                    "type": "Feedback loop node",
                    "rationale": f"Breaking this node could disrupt the feedback loop: {loop.get('description', '')}",
                    "potential_impact": "Medium"
                })
        
        # Deduplicate interventions
        unique_interventions = []
        seen_nodes = set()
        
        for intervention in interventions:
            node = intervention.get("node")
            if node and node not in seen_nodes:
                seen_nodes.add(node)
                unique_interventions.append(intervention)
        
        return unique_interventions
    
    def _generate_synthesis(self, question, entities, relationships, network_analysis, interventions, economic_data=None, geopolitical_data=None):
        """
        Generate a synthesis of the causal network analysis.
        
        Args:
            question: The analytical question
            entities: List of entity dictionaries
            relationships: List of relationship dictionaries
            network_analysis: Dictionary containing network analysis results
            interventions: List of intervention dictionaries
            
        Returns:
            Dictionary containing the synthesis
        """
        logger.info("Generating synthesis of causal network analysis...")


        
        prompt = f"""
        Synthesize the following causal network analysis for the question:
        
        "{question}"
        
        Entities:
        {json.dumps(entities, indent=2)}
        
        Causal Relationships:
        {json.dumps(relationships, indent=2)}
        
        Network Analysis:
        - Central Nodes: {json.dumps(network_analysis.get("central_nodes", []), indent=2)}
        - End Nodes: {json.dumps(network_analysis.get("end_nodes", []), indent=2)}
        - Feedback Loops: {json.dumps(network_analysis.get("feedback_loops", []), indent=2)}
        
        Potential Intervention Points:
        {json.dumps(interventions, indent=2)}
        
        """
        if economic_data:
            prompt += f"""
            Relevant Economic Data:
            {json.dumps(economic_data, indent=2)}
            """

        if geopolitical_data:
            prompt += f"""
            Relevant Geopolitical Data:
            {json.dumps(geopolitical_data, indent=2)}
            """
        prompt += f"""        Based on this causal network analysis:
        1. What are the key drivers in this system?
        2. What are the most important causal pathways?
        3. What feedback dynamics are present?
        4. Where are the most effective intervention points?
        
        Provide:
        1. A final judgment about the causal dynamics
        2. A rationale for this judgment
        3. A confidence level in this judgment (High/Medium/Low)
        4. Potential biases that might be affecting this analysis
        
        Return as JSON:
        {{
            "final_judgment": "Your assessment of the causal dynamics",
            "judgment_rationale": "Explanation for your judgment",
            "key_drivers": ["Driver 1", "Driver 2", ...],
            "key_pathways": ["Pathway 1", "Pathway 2", ...],
            "recommended_interventions": ["Intervention 1", "Intervention 2", ...],
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
                    "key_drivers": ["Error in synthesis generation"],
                    "key_pathways": ["Error in synthesis generation"],
                    "recommended_interventions": ["Error in synthesis generation"],
                    "confidence_level": "Low",
                    "potential_biases": ["Technical error bias"]
                }
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing synthesis: {e}")
            return {
                "final_judgment": f"Error generating synthesis: {str(e)}",
                "judgment_rationale": "Error in synthesis generation",
                "key_drivers": ["Error in synthesis generation"],
                "key_pathways": ["Error in synthesis generation"],
                "recommended_interventions": ["Error in synthesis generation"],
                "confidence_level": "Low",
                "potential_biases": ["Technical error bias"]
            }

    def _fetch_economic_data(self, question):
        """
        Fetch relevant economic data based on the question.

        Args:
            question: The analytical question

        Returns:
            A dictionary containing relevant economic data or None if data fetching fails.
        """
        logger.info("Fetching economic data...")
        try:
            economics_mcp = self.mcp_registry.get_mcp("economics_mcp")
            if economics_mcp:
                # Determine relevant indicators based on the question
                if "inflation" in question.lower():
                    indicators = ["CPIAUCSL", "FPCPITOTLZGUSA"]  # CPI, Inflation
                elif "unemployment" in question.lower():
                    indicators = ["UNRATE"]  # Unemployment Rate
                elif "gdp" in question.lower():
                    indicators = ["GDP"]
                else:
                    indicators = ["CPIAUCSL", "UNRATE", "GDP"]

                # Fetch data for each indicator
                economic_data = {}
                for indicator in indicators:
                    series = economics_mcp.get_fred_data(indicator, "2020-01-01", "2023-12-31")
                    economic_data[indicator] = series
                return economic_data
            else:
                return None
        except Exception as e:
            logger.error(f"Error fetching economic data: {e}")
            return None

    def _fetch_geopolitical_data(self, question):
        """
        Fetch relevant geopolitical event data based on the question.
        """
        try:
            geopolitics_mcp = self.mcp_registry.get_mcp("geopolitics_mcp")
            if geopolitics_mcp:
                # Fetch data for the relevant location and date range
                return geopolitics_mcp.get_gdelt_data(location="Global", start_date="20230101", end_date="20231231")
        except Exception as e:
            logger.error(f"Error fetching geopolitical data: {e}")
            return None
