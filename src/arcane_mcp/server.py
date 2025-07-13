"""
ARCANE MCP Server - Main Server Implementation
"""

import asyncio
import logging
import json
import os
import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server
 
from arcane_mcp.core.id_resolver import IdentifierResolver
from arcane_mcp.data_sources.arxiv_client import ArxivClient
from arcane_mcp.data_sources.semanticScholar_client import SemanticScholarClient  
from arcane_mcp.data_sources.openCitations_client import OpenCitationsClient
from arcane_mcp.config.config_manager import config_manager
from arcane_mcp.agent.orchestrator import CrossDomainResearchAgent

import weave

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arcane")

class AcademicDiscoveryServer:
    """Main MCP server for academic paper discovery and analysis"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Use the new configuration manager
        self.config = config_manager
        
        # Initialize server
        self.server = Server("arcane")
        
        # Initialize API clients first
        self.clients = {}
        self._init_clients()
        
        # Initialize core components with configured clients
        self.identifier_resolver = IdentifierResolver(
            db_path=self.config.get('main', 'database.path', 'academic_papers.db'),
            arxiv_client=self.clients['arxiv'],
            semantic_scholar_client=self.clients['semantic_scholar'],
            opencitations_client=self.clients['opencitations']
        )
        
        # Initialize cross-domain research agent
        agent_clients = {
            'arxiv': self.clients['arxiv'],
            'semantic_scholar': self.clients['semantic_scholar'],
            'opencitations': self.clients['opencitations'],
            'identifier_resolver': self.identifier_resolver
        }
        self.research_agent = CrossDomainResearchAgent(agent_clients)
        
        # Register handlers
        self._register_handlers()

        # Initialize Weave monitoring
        weave.init("ARCANE")
        
        logger.info(" MCP Server initialized")
    
    def _init_clients(self):
        """Initialize API clients with configuration"""
        self.clients['arxiv'] = ArxivClient()
        
        # Use environment variables for API keys (not stored in config files for security)
        import os
        s2_api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
        self.clients['semantic_scholar'] = SemanticScholarClient(api_key=s2_api_key)
        
        oc_token = os.getenv('OPENCITATIONS_ACCESS_TOKEN')
        self.clients['opencitations'] = OpenCitationsClient(access_token=oc_token)
    
    def _register_handlers(self):
        """Register all MCP handlers"""
        self.server.list_tools()(self.list_tools)
        self.server.call_tool()(self.call_tool)
        self.server.list_resources()(self.list_resources)
        self.server.read_resource()(self.read_resource)
        self.server.list_prompts()(self.list_prompts)
        self.server.get_prompt()(self.get_prompt)
    
    async def list_tools(self) -> List[types.Tool]:
        """List available tools"""
        return [
            types.Tool(
                name="search_papers",
                description="Search for academic papers across arXiv, Semantic Scholar, and other sources",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "sources": {
                            "type": "array", 
                            "items": {"type": "string", "enum": ["arxiv", "semantic_scholar", "all"]},
                            "default": ["all"]
                        },
                        "max_results": {"type": "integer", "default": 10}
                    },
                    "required": ["query"]
                }
            ),
            
            types.Tool(
                name="get_paper",
                description="Get detailed information about a specific paper. Supports: arXiv IDs (e.g., 2006.12469), DOIs (e.g., 10.1088/2632-2153/ac362b), and Semantic Scholar IDs",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "identifier": {"type": "string", "description": "Paper identifier (arXiv ID, DOI, or Semantic Scholar ID)"},
                        "include_citations": {"type": "boolean", "default": False}
                    },
                    "required": ["identifier"]
                }
            ),
            
            types.Tool(
                name="resolve_identifiers",
                description="Find all known identifiers for a paper. Supports: arXiv IDs, DOIs, and Semantic Scholar IDs. Returns all available identifiers for the same paper across different databases.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "identifier": {"type": "string", "description": "Paper identifier (arXiv ID, DOI, or Semantic Scholar ID)"}
                    },
                    "required": ["identifier"]
                }
            ),
            
            types.Tool(
                name="get_citations",
                description="Get citation data for a paper. Works best with DOI identifiers. Returns incoming citations and outgoing references from OpenCitations and Semantic Scholar.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "identifier": {"type": "string", "description": "Paper identifier (DOI recommended for best results)"},
                        "direction": {"type": "string", "enum": ["incoming", "outgoing", "both"], "default": "both"}
                    },
                    "required": ["identifier"]
                }
            ),
            
            types.Tool(
                name="build_citation_graph",
                description="Build a citation network graph around a paper. Requires DOI identifier for OpenCitations data. Generates network visualization data with nodes and edges.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "identifier": {"type": "string", "description": "Paper identifier (DOI required)"},
                        "depth": {"type": "integer", "default": 2, "minimum": 1, "maximum": 3},
                        "max_nodes": {"type": "integer", "default": 50}
                    },
                    "required": ["identifier"]
                }
            ),
            
            # Cross-Domain Research Agent Tools
            types.Tool(
                name="analyze_research_problem",
                description="Analyze a research problem and discover cross-domain solutions by abstracting mathematical patterns, searching across scientific domains, and translating solutions back to the target domain. Provides comprehensive analysis with full reasoning trace.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "problem_description": {
                            "type": "string", 
                            "description": "Natural language description of the research problem to analyze"
                        },
                        "max_solutions": {
                            "type": "integer", 
                            "default": 10, 
                            "minimum": 1, 
                            "maximum": 20,
                            "description": "Maximum number of cross-domain solutions to discover"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session ID for tracking (auto-generated if not provided)"
                        }
                    },
                    "required": ["problem_description"]
                }
            ),
            
            types.Tool(
                name="quick_research_analysis", 
                description="Perform a quick cross-domain research analysis with limited depth for faster results. Good for initial exploration or when time is limited.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "problem_description": {
                            "type": "string",
                            "description": "Research problem description"
                        },
                        "max_results": {
                            "type": "integer",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Maximum number of solutions for quick analysis"
                        }
                    },
                    "required": ["problem_description"]
                }
            ),
            
            types.Tool(
                name="abstract_mathematical_patterns",
                description="Extract mathematical and methodological patterns from a research problem description. Useful for understanding the core structure of a problem before cross-domain search.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "problem_description": {
                            "type": "string",
                            "description": "Research problem description to abstract"
                        }
                    },
                    "required": ["problem_description"]
                }
            ),
            
            types.Tool(
                name="discover_cross_domain_solutions",
                description="Search for structurally similar solutions across scientific domains using abstracted mathematical patterns. Requires prior problem abstraction.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "mathematical_patterns": {
                            "type": "object",
                            "description": "Abstracted mathematical patterns from problem_abstractor"
                        },
                        "max_results": {
                            "type": "integer",
                            "default": 15,
                            "description": "Maximum number of cross-domain papers to find"
                        }
                    },
                    "required": ["mathematical_patterns"]
                }
            ),
            
            types.Tool(
                name="translate_solutions",
                description="Translate discovered cross-domain solutions back to the target domain with concept mappings, methodology adaptations, and implementation guidance.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "search_results": {
                            "type": "object", 
                            "description": "Cross-domain search results to translate"
                        }
                    },
                    "required": ["search_results"]
                }
            )
        ]
    
    @weave.op()
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle tool calls"""
        try:
            if name == "search_papers":
                result = await self._search_papers(**arguments)
            elif name == "get_paper":
                result = await self._get_paper(**arguments)
            elif name == "resolve_identifiers":
                result = await self._resolve_identifiers(**arguments)
            elif name == "get_citations":
                result = await self._get_citations(**arguments)
            elif name == "build_citation_graph":
                result = await self._build_citation_graph(**arguments)
            # Cross-Domain Research Agent Tools
            elif name == "analyze_research_problem":
                result = await self._analyze_research_problem(**arguments)
            elif name == "quick_research_analysis":
                result = await self._quick_research_analysis(**arguments)
            elif name == "abstract_mathematical_patterns":
                result = await self._abstract_mathematical_patterns(**arguments)
            elif name == "discover_cross_domain_solutions":
                result = await self._discover_cross_domain_solutions(**arguments)
            elif name == "translate_solutions":
                result = await self._translate_solutions(**arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
            
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]
            
        except Exception as e:
            logger.error(f"Error in tool {name}: {str(e)}")
            return [types.TextContent(
                type="text", 
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    
    # Tool implementations
    async def _search_papers(self, query: str, sources: List[str] = ["all"], 
                           max_results: int = 10) -> Dict[str, Any]:
        """Search for papers across multiple sources"""
        if "all" in sources:
            sources = ["arxiv", "semantic_scholar"]
        
        results = {"query": query, "sources": {}, "unified_results": []}
        
        for source in sources:
            if source == "arxiv":
                async with self.clients['arxiv'] as client:
                    papers = await client.search_papers(query=query, max_results=max_results)
                    results["sources"]["arxiv"] = papers
            
            elif source == "semantic_scholar":
                async with self.clients['semantic_scholar'] as client:
                    papers = await client.search_papers(query=query, limit=max_results)
                    results["sources"]["semantic_scholar"] = papers
        
        # Unify results
        unified_papers = []
        seen_unified_ids = set()
        
        for source, papers in results["sources"].items():
            for paper in papers:
                identifier = self._extract_primary_identifier(paper, source)
                if identifier:
                    unified_paper = await self.identifier_resolver.resolve_paper(identifier, source)
                    if unified_paper and unified_paper.unified_id not in seen_unified_ids:
                        enriched_paper = self._enrich_unified_paper(unified_paper, paper, source)
                        unified_papers.append(enriched_paper)
                        seen_unified_ids.add(unified_paper.unified_id)
        
        results["unified_results"] = unified_papers
        results["total_unified"] = len(unified_papers)
        
        return results
    
    async def _get_paper(self, identifier: str, include_citations: bool = False) -> Dict[str, Any]:
        """Get detailed paper information"""
        try:
            # Try to determine source hint from identifier
            source_hint = None
            if identifier.startswith('10.') or identifier.startswith('doi:'):
                source_hint = 'semantic_scholar'
            elif identifier.replace('.', '').replace('v', '').replace('-', '').isdigit():
                source_hint = 'arxiv'
            
            unified_paper = await self.identifier_resolver.resolve_paper(identifier, source_hint)
            if not unified_paper:
                return {"error": f"Paper not found: {identifier}"}
            
            result = {
                "identifiers": unified_paper.identifiers,
                "metadata": unified_paper.canonical_metadata,
                "confidence_score": unified_paper.confidence_score
            }
            
            if include_citations:
                citation_data = await self._get_citation_data(unified_paper)
                if citation_data:
                    result["citation_data"] = citation_data
            
            return result
        except Exception as e:
            logger.error(f"Error in _get_paper: {str(e)}")
            return {"error": f"Error retrieving paper: {str(e)}"}
    
    async def _resolve_identifiers(self, identifier: str) -> Dict[str, Any]:
        """Resolve all identifiers for a paper"""
        try:
            # Try to determine source hint from identifier
            source_hint = None
            if identifier.startswith('10.') or identifier.startswith('doi:'):
                source_hint = 'semantic_scholar'
            elif identifier.replace('.', '').replace('v', '').replace('-', '').isdigit():
                source_hint = 'arxiv'
            
            unified_paper = await self.identifier_resolver.resolve_paper(identifier, source_hint)
            if not unified_paper:
                return {"error": f"Could not resolve identifier: {identifier}"}
            
            return {
                "identifiers": unified_paper.identifiers,
                "canonical_metadata": unified_paper.canonical_metadata,
                "confidence_score": unified_paper.confidence_score
            }
        except Exception as e:
            logger.error(f"Error in _resolve_identifiers: {str(e)}")
            return {"error": f"Error resolving identifiers: {str(e)}"}
    
    async def _get_citations(self, identifier: str, direction: str = "both") -> Dict[str, Any]:
        """Get citation data for a paper"""
        try:
            # Try to determine source hint from identifier
            source_hint = None
            if identifier.startswith('10.') or identifier.startswith('doi:'):
                source_hint = 'semantic_scholar'
            elif identifier.replace('.', '').replace('v', '').replace('-', '').isdigit():
                source_hint = 'arxiv'
            
            unified_paper = await self.identifier_resolver.resolve_paper(identifier, source_hint)
            if not unified_paper:
                return {"error": f"Paper not found: {identifier}"}
            
            citation_data = {"paper": unified_paper.canonical_metadata}
            
            if unified_paper.identifiers.get('doi'):
                async with self.clients['opencitations'] as client:
                    oc_citations = await client.get_citations(unified_paper.identifiers['doi'])
                    citation_data["opencitations"] = oc_citations
            
            if unified_paper.identifiers.get('semantic_scholar'):
                async with self.clients['semantic_scholar'] as client:
                    if direction in ["incoming", "both"]:
                        citations = await client.get_paper_citations(
                            unified_paper.identifiers['semantic_scholar']
                        )
                        citation_data["citing_papers"] = citations
                    
                    if direction in ["outgoing", "both"]:
                        references = await client.get_paper_references(
                            unified_paper.identifiers['semantic_scholar']
                        )
                        citation_data["referenced_papers"] = references
            
            return citation_data
        except Exception as e:
            logger.error(f"Error in _get_citations: {str(e)}")
            return {"error": f"Error retrieving citations: {str(e)}"}
    
    async def _build_citation_graph(self, identifier: str, depth: int = 2, 
                                  max_nodes: int = 50) -> Dict[str, Any]:
        """Build citation network graph"""
        try:
            # Try to determine source hint from identifier
            source_hint = None
            if identifier.startswith('10.') or identifier.startswith('doi:'):
                source_hint = 'semantic_scholar'
            elif identifier.replace('.', '').replace('v', '').replace('-', '').isdigit():
                source_hint = 'arxiv'
            
            unified_paper = await self.identifier_resolver.resolve_paper(identifier, source_hint)
            if not unified_paper:
                return {"error": f"Paper not found: {identifier}"}
            
            if unified_paper.identifiers.get('doi'):
                async with self.clients['opencitations'] as client:
                    graph = await client.build_citation_graph(
                        unified_paper.identifiers['doi'], 
                        depth=depth
                    )
                    return graph
            
            return {"error": "Citation graph requires DOI identifier"}
        except Exception as e:
            logger.error(f"Error in _build_citation_graph: {str(e)}")
            return {"error": f"Error building citation graph: {str(e)}"}
    
    # Cross-Domain Research Agent Tools Implementation
    async def _analyze_research_problem(self, problem_description: str, 
                                      max_solutions: int = 10,
                                      session_id: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive cross-domain research problem analysis"""
        try:
            agent_result = await self.research_agent.analyze_research_problem(
                problem_description, max_solutions, session_id
            )
            
            # Convert result to serializable format
            result = {
                "session_id": agent_result.session_id,
                "original_problem": agent_result.original_problem,
                "problem_analysis": {
                    "domain": agent_result.abstracted_problem.domain_context.primary_domain,
                    "problem_types": [pt.value for pt in agent_result.abstracted_problem.mathematical_pattern.problem_types],
                    "mathematical_structures": [ms.value for ms in agent_result.abstracted_problem.mathematical_pattern.mathematical_structures],
                    "abstraction_confidence": agent_result.abstracted_problem.abstraction_confidence,
                    "domain_agnostic_terms": agent_result.abstracted_problem.mathematical_pattern.domain_agnostic_terms
                },
                "cross_domain_matches": [
                    {
                        "paper_id": match.paper_id,
                        "title": match.title,
                        "authors": match.authors,
                        "domain": match.domain,
                        "year": match.year,
                        "citation_count": match.citation_count,
                        "similarity_scores": {
                            "overall": match.similarity_score.overall_similarity,
                            "mathematical": match.similarity_score.mathematical_similarity,
                            "methodological": match.similarity_score.methodological_similarity
                        },
                        "matching_patterns": match.matching_patterns,
                        "relevance_explanation": match.relevance_explanation
                    }
                    for match in agent_result.search_result.matches
                ],
                "translated_solutions": [
                    {
                        "source_paper_title": solution.source_paper.title,
                        "source_domain": solution.source_paper.domain,
                        "target_domain": solution.target_domain,
                        "methodology_translation": {
                            "original_method": solution.methodology_translation.original_method,
                            "translated_method": solution.methodology_translation.translated_method,
                            "adaptation_steps": solution.methodology_translation.adaptation_steps,
                            "implementation_notes": solution.methodology_translation.implementation_notes
                        },
                        "concept_mappings": [
                            {
                                "source": mapping.source_concept,
                                "target": mapping.target_concept,
                                "confidence": mapping.confidence
                            }
                            for mapping in solution.concept_mappings
                        ],
                        "mathematical_adaptations": solution.mathematical_adaptations,
                        "implementation_roadmap": solution.implementation_roadmap,
                        "feasibility_score": solution.feasibility_score,
                        "expected_benefits": solution.expected_benefits,
                        "potential_limitations": solution.potential_limitations
                    }
                    for solution in agent_result.translation_result.translated_solutions
                ],
                "recommendations": agent_result.recommendations,
                "execution_summary": agent_result.execution_summary,
                "reasoning_trace": {
                    "total_steps": len(agent_result.process_trace.reasoning_steps),
                    "decision_points": len(agent_result.process_trace.decision_points),
                    "key_insights": agent_result.process_trace.key_insights,
                    "success_metrics": agent_result.process_trace.success_metrics
                }
            }
            
            # Convert numpy types to native Python types
            return self._convert_numpy_types(result)
            
        except Exception as e:
            logger.error(f"Error in comprehensive research analysis: {str(e)}")
            return {"error": f"Error in research analysis: {str(e)}"}
    
    async def _quick_research_analysis(self, problem_description: str, 
                                     max_results: int = 5) -> Dict[str, Any]:
        """Quick cross-domain research analysis"""
        try:
            result = await self.research_agent.quick_analysis(problem_description, max_results)
            return self._convert_numpy_types(result)
        except Exception as e:
            logger.error(f"Error in quick research analysis: {str(e)}")
            return {"error": f"Error in quick analysis: {str(e)}"}
    
    async def _abstract_mathematical_patterns(self, problem_description: str) -> Dict[str, Any]:
        """Extract mathematical patterns from research problem"""
        try:
            abstracted_problem = await self.research_agent.problem_abstractor.abstract_problem(problem_description)
            
            result = {
                "problem_types": [pt.value for pt in abstracted_problem.mathematical_pattern.problem_types],
                "mathematical_structures": [ms.value for ms in abstracted_problem.mathematical_pattern.mathematical_structures],
                "domain": abstracted_problem.domain_context.primary_domain,
                "confidence": abstracted_problem.abstraction_confidence,
                "domain_agnostic_terms": abstracted_problem.mathematical_pattern.domain_agnostic_terms,
                "methodological_keywords": abstracted_problem.mathematical_pattern.methodological_keywords,
                "iterative_process": abstracted_problem.mathematical_pattern.iterative_process,
                "uncertainty_sources": abstracted_problem.mathematical_pattern.uncertainty_sources,
                "objective_function": abstracted_problem.mathematical_pattern.objective_function,
                "constraints": abstracted_problem.mathematical_pattern.constraints,
                # Note: Full abstracted problem data is not serializable for JSON output
                # Use the 'analyze_research_problem' tool for complete workflow with all data
            }
            
            # Convert numpy types to native Python types
            return self._convert_numpy_types(result)
            
        except Exception as e:
            logger.error(f"Error in mathematical pattern abstraction: {str(e)}")
            return {"error": f"Error in pattern abstraction: {str(e)}"}
    
    async def _discover_cross_domain_solutions(self, mathematical_patterns: Dict[str, Any], 
                                             max_results: int = 15) -> Dict[str, Any]:
        """Search for cross-domain solutions using mathematical patterns"""
        try:
            # This would require reconstructing the AbstractedProblem from the patterns
            # For now, return guidance to use the full analyze_research_problem tool
            return {
                "message": "For cross-domain solution discovery, please use the 'analyze_research_problem' tool which provides the complete workflow including pattern abstraction, cross-domain search, and solution translation.",
                "recommendation": "Use analyze_research_problem with your original problem description for best results."
            }
        except Exception as e:
            logger.error(f"Error in cross-domain solution discovery: {str(e)}")
            return {"error": f"Error in solution discovery: {str(e)}"}
    
    async def _translate_solutions(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Translate cross-domain solutions to target domain"""
        try:
            # This would require reconstructing SearchResult objects
            # For now, return guidance to use the full analyze_research_problem tool
            return {
                "message": "For solution translation, please use the 'analyze_research_problem' tool which provides the complete workflow including translation with concept mappings and implementation guidance.",
                "recommendation": "Use analyze_research_problem with your original problem description for complete translation results."
            }
        except Exception as e:
            logger.error(f"Error in solution translation: {str(e)}")
            return {"error": f"Error in solution translation: {str(e)}"}
    
    # Helper methods
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def _extract_primary_identifier(self, paper: Dict[str, Any], source: str) -> Optional[str]:
        """Extract the primary identifier from a paper"""
        if source == "arxiv":
            return paper.get('identifier')
        elif source == "semantic_scholar":
            return paper.get('identifier') or paper.get('paperId')
        return None
    
    def _enrich_unified_paper(self, unified_paper, source_paper: Dict[str, Any], source: str):
        """Enrich unified paper with source-specific data"""
        enriched = {
            "identifiers": unified_paper.identifiers,
            "metadata": unified_paper.canonical_metadata,
            "confidence_score": unified_paper.confidence_score,
            "source_data": {source: source_paper}
        }
        
        if source == "semantic_scholar":
            enriched["metrics"] = {
                "citation_count": source_paper.get('citation_count', 0),
                "reference_count": source_paper.get('reference_count', 0)
            }
        
        return enriched
    
    async def _get_citation_data(self, unified_paper):
        """Get citation data for a unified paper"""
        citation_data = {}
        
        if unified_paper.identifiers.get('doi'):
            async with self.clients['opencitations'] as client:
                oc_data = await client.get_citations(unified_paper.identifiers['doi'])
                citation_data["opencitations"] = oc_data
        
        return citation_data if citation_data else None
    
    # Resources and Prompts (simplified for now)
    async def list_resources(self) -> List[types.Resource]:
        return []
    
    async def read_resource(self, uri: str) -> str:
        return "{}"
    
    async def list_prompts(self) -> List[types.Prompt]:
        return [
            types.Prompt(
                name="comprehensive-paper-analysis",
                description="Analyze a research paper comprehensively",
                arguments=[
                    types.PromptArgument(
                        name="paper_identifier",
                        description="Paper identifier",
                        required=True
                    )
                ]
            )
        ]
    
    async def get_prompt(self, name: str, arguments: Dict[str, str]) -> types.GetPromptResult:
        if name == "comprehensive-paper-analysis":
            paper_id = arguments.get("paper_identifier")
            content = f"""
Analyze the paper "{paper_id}" comprehensively:

1. Use `resolve_identifiers` to find all IDs
2. Use `get_paper` with citations enabled
3. Use `get_citations` for impact analysis
4. Use `build_citation_graph` for network context

Provide insights on the paper's contribution, impact, and context.
"""
            return types.GetPromptResult(
                description="Comprehensive paper analysis",
                messages=[types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=content)
                )]
            )
        
        raise ValueError(f"Unknown prompt: {name}")


def main():
    """Main entry point for the MCP server"""
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
        import traceback
        traceback.print_exc()

async def _main():
    """Async main entry point"""
    server = AcademicDiscoveryServer()
    
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            server.server.create_initialization_options()
        )

if __name__ == "__main__":
    main()