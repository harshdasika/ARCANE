"""
ARCANE MCP Server - Main Server Implementation
"""

import asyncio
import logging
import json
import os
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
from arcane_mcp.config.settings import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arcane")

class AcademicDiscoveryServer:
    """Main MCP server for academic paper discovery and analysis"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize server
        self.server = Server("arcane")
        
        # Initialize API clients first
        self.clients = {}
        self._init_clients()
        
        # Initialize core components with configured clients
        self.identifier_resolver = IdentifierResolver(
            db_path=self.config.get('database_path', 'academic_papers.db'),
            arxiv_client=self.clients['arxiv'],
            semantic_scholar_client=self.clients['semantic_scholar'],
            opencitations_client=self.clients['opencitations']
        )
        
        # Register handlers
        self._register_handlers()
        
        logger.info(" MCP Server initialized")
    
    def _init_clients(self):
        """Initialize API clients with configuration"""
        self.clients['arxiv'] = ArxivClient()
        
        s2_api_key = self.config.get('semantic_scholar_api_key')
        self.clients['semantic_scholar'] = SemanticScholarClient(api_key=s2_api_key)
        
        oc_token = self.config.get('opencitations_access_token')
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
            )
        ]
    
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
    
    # Helper methods
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