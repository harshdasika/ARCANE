"""
Core Identifier Resolver for ARCANE MCP Server
"""

import hashlib
import asyncio
import logging
import sqlite3
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from fuzzywuzzy import fuzz
from pathlib import Path

# Import API clients
from arcane_mcp.data_sources.arxiv_client import ArxivClient
from arcane_mcp.data_sources.semanticScholar_client import SemanticScholarClient
from arcane_mcp.data_sources.openCitations_client import OpenCitationsClient

class IdentifierType(Enum):
    DOI = "doi"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    PMID = "pmid"
    ISBN = "isbn"
    UNKNOWN = "unknown"

@dataclass
class PaperIdentifier:
    value: str
    type: IdentifierType
    confidence: float = 1.0

@dataclass
class UnifiedPaper:
    unified_id: str
    identifiers: Dict[str, str]
    canonical_metadata: Dict[str, Any]
    versions: List[Dict[str, Any]]
    citation_data: Dict[str, Any] = None
    confidence_score: float = 1.0

class IdentifierResolver:
    """Core identifier resolution system"""
    
    def __init__(self, db_path: str = "paper_mappings.db", 
                 arxiv_client: Optional[ArxivClient] = None,
                 semantic_scholar_client: Optional[SemanticScholarClient] = None,
                 opencitations_client: Optional[OpenCitationsClient] = None):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
        
        # Initialize API clients (use provided ones or create defaults)
        self.arxiv_client = arxiv_client or ArxivClient()
        self.semantic_scholar_client = semantic_scholar_client or SemanticScholarClient()
        self.opencitations_client = opencitations_client or OpenCitationsClient()
        
    def _init_database(self):
        """Initialize SQLite database"""
        # Create directory if it doesn't exist
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_mappings (
                unified_id TEXT PRIMARY KEY,
                doi TEXT,
                arxiv_id TEXT,
                semantic_scholar_id TEXT,
                pmid TEXT,
                isbn TEXT,
                canonical_metadata TEXT,
                versions TEXT,
                citation_data TEXT,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doi ON paper_mappings(doi)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_arxiv ON paper_mappings(arxiv_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ss ON paper_mappings(semantic_scholar_id)")
        
        conn.commit()
        conn.close()
    
    def _generate_unified_id(self, canonical_metadata: Dict[str, Any]) -> str:
        """Generate stable unified ID"""
        title = canonical_metadata.get('title', '').lower().strip()
        authors = canonical_metadata.get('authors', [])
        year = str(canonical_metadata.get('year', ''))
        
        author_string = '|'.join(sorted([str(author).lower().strip() for author in authors]))
        content = f"{title}|{author_string}|{year}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _normalize_identifier(self, identifier: str) -> PaperIdentifier:
        """Detect and normalize identifier types"""
        identifier = identifier.strip()
        
        # DOI patterns
        if identifier.startswith('10.') or identifier.startswith('doi:'):
            clean_doi = identifier.replace('doi:', '')
            return PaperIdentifier(clean_doi, IdentifierType.DOI)
        
        # arXiv patterns - improved detection
        # Check for arXiv format: YYMM.NNNNN or YYMM.NNNNNvN
        if (len(identifier) >= 9 and 
            identifier[:2].isdigit() and 
            identifier[2:4].isdigit() and 
            identifier[4] == '.' and 
            identifier[5:].replace('v', '').isdigit()):
            clean_arxiv = identifier.replace('arxiv:', '').replace('arXiv:', '')
            return PaperIdentifier(clean_arxiv, IdentifierType.ARXIV)
        
        # Also check for arXiv in identifier
        if 'arxiv' in identifier.lower():
            clean_arxiv = identifier.replace('arxiv:', '').replace('arXiv:', '')
            return PaperIdentifier(clean_arxiv, IdentifierType.ARXIV)
        
        # Semantic Scholar numeric ID
        if identifier.isdigit() and len(identifier) >= 6:
            return PaperIdentifier(identifier, IdentifierType.SEMANTIC_SCHOLAR)
        
        # PubMed ID
        if identifier.startswith('PMID:') or (identifier.isdigit() and len(identifier) <= 8):
            clean_pmid = identifier.replace('PMID:', '')
            return PaperIdentifier(clean_pmid, IdentifierType.PMID)
        
        return PaperIdentifier(identifier, IdentifierType.UNKNOWN)
    
    async def resolve_paper(self, identifier: str, source_hint: Optional[str] = None) -> Optional[UnifiedPaper]:
        """Main resolution method"""
        try:
            paper_id = self._normalize_identifier(identifier)
            
            # Check cache first
            cached_paper = await self._lookup_cached_mapping(paper_id)
            if cached_paper:
                self.logger.info(f"Found cached mapping for {identifier}")
                return cached_paper
            
            # Create real unified paper using API calls
            unified_paper = await self._create_real_unified_paper(paper_id)
            
            if unified_paper:
                await self._cache_mapping(unified_paper)
                self.logger.info(f"Created mapping for {identifier} -> {unified_paper.unified_id}")
                return unified_paper
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error resolving {identifier}: {str(e)}")
            return None
    
    async def _lookup_cached_mapping(self, paper_id: PaperIdentifier) -> Optional[UnifiedPaper]:
        """Look up existing mapping in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        column_map = {
            IdentifierType.DOI: 'doi',
            IdentifierType.ARXIV: 'arxiv_id',
            IdentifierType.SEMANTIC_SCHOLAR: 'semantic_scholar_id',
            IdentifierType.PMID: 'pmid'
        }
        
        if paper_id.type in column_map:
            column = column_map[paper_id.type]
            cursor.execute(f"SELECT * FROM paper_mappings WHERE {column} = ?", (paper_id.value,))
            row = cursor.fetchone()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                data = dict(zip(columns, row))
                
                unified_paper = UnifiedPaper(
                    unified_id=data['unified_id'],
                    identifiers={
                        'doi': data.get('doi'),
                        'arxiv': data.get('arxiv_id'),
                        'semantic_scholar': data.get('semantic_scholar_id'),
                        'pmid': data.get('pmid')
                    },
                    canonical_metadata=json.loads(data['canonical_metadata']) if data['canonical_metadata'] else {},
                    versions=json.loads(data['versions']) if data['versions'] else [],
                    citation_data=json.loads(data['citation_data']) if data['citation_data'] else None,
                    confidence_score=data.get('confidence_score', 1.0)
                )
                
                # Clean up None values
                unified_paper.identifiers = {k: v for k, v in unified_paper.identifiers.items() if v is not None}
                
                conn.close()
                return unified_paper
        
        conn.close()
        return None
    
    async def _create_real_unified_paper(self, paper_id: PaperIdentifier) -> Optional[UnifiedPaper]:
        """Create unified paper using real API calls"""
        
        self.logger.info(f"Creating real unified paper for {paper_id.value} (type: {paper_id.type})")
        
        paper_data = None
        source = None
        
        # Try to get paper data based on identifier type
        if paper_id.type == IdentifierType.ARXIV:
            self.logger.info(f"Fetching from arXiv: {paper_id.value}")
            try:
                async with self.arxiv_client as client:
                    paper_data = await client.get_paper(paper_id.value)
                    source = 'arxiv'
                    self.logger.info(f"arXiv result: {paper_data is not None}")
                    if paper_data:
                        self.logger.info(f"arXiv title: {paper_data.get('title', 'No title')[:50]}...")
            except Exception as e:
                self.logger.error(f"Error fetching from arXiv: {e}")
                
        elif paper_id.type == IdentifierType.DOI:
            # Try Semantic Scholar first (better data)
            self.logger.info(f"Fetching from Semantic Scholar: {paper_id.value}")
            try:
                async with self.semantic_scholar_client as client:
                    paper_data = await client.get_paper(paper_id.value)
                    source = 'semantic_scholar'
                    self.logger.info(f"Semantic Scholar result: {paper_data is not None}")
                    if paper_data:
                        self.logger.info(f"Semantic Scholar title: {paper_data.get('title', 'No title')[:50]}...")
            except Exception as e:
                self.logger.error(f"Error fetching from Semantic Scholar: {e}")
            
            # Fallback to OpenCitations if Semantic Scholar fails
            if not paper_data:
                self.logger.info(f"Trying OpenCitations: {paper_id.value}")
                try:
                    async with self.opencitations_client as client:
                        paper_data = await client.get_paper(paper_id.value)
                        source = 'opencitations'
                        self.logger.info(f"OpenCitations result: {paper_data is not None}")
                        if paper_data:
                            self.logger.info(f"OpenCitations title: {paper_data.get('title', 'No title')[:50]}...")
                except Exception as e:
                    self.logger.error(f"Error fetching from OpenCitations: {e}")
                    
        elif paper_id.type == IdentifierType.SEMANTIC_SCHOLAR:
            self.logger.info(f"Fetching from Semantic Scholar: {paper_id.value}")
            try:
                async with self.semantic_scholar_client as client:
                    paper_data = await client.get_paper(paper_id.value)
                    source = 'semantic_scholar'
                    self.logger.info(f"Semantic Scholar result: {paper_data is not None}")
                    if paper_data:
                        self.logger.info(f"Semantic Scholar title: {paper_data.get('title', 'No title')[:50]}...")
            except Exception as e:
                self.logger.error(f"Error fetching from Semantic Scholar: {e}")
        
        if not paper_data:
            self.logger.warning(f"No paper data found for {paper_id.value}")
            return None
        
        self.logger.info(f"Found paper data: {paper_data.get('title', 'No title')[:50]}...")
        
        # Normalize paper data to canonical format
        canonical_metadata = self._normalize_to_canonical(paper_data, source)
        
        # Generate unified ID
        unified_id = self._generate_unified_id(canonical_metadata)
        
        # Build identifiers dictionary
        identifiers = {}
        if paper_id.type == IdentifierType.ARXIV:
            identifiers['arxiv'] = paper_id.value
        elif paper_id.type == IdentifierType.DOI:
            identifiers['doi'] = paper_id.value
        elif paper_id.type == IdentifierType.SEMANTIC_SCHOLAR:
            identifiers['semantic_scholar'] = paper_id.value
        
        # Add external IDs if available
        if 'external_ids' in paper_data:
            external_ids = paper_data['external_ids']
            if external_ids.get('DOI'):
                identifiers['doi'] = external_ids['DOI']
            if external_ids.get('ArXiv'):
                identifiers['arxiv'] = external_ids['ArXiv']
        
        # Create version info
        versions = [{
            'identifier': paper_id.value,
            'source': source,
            'retrieved_at': '2024-01-01T00:00:00Z'
        }]
        
        self.logger.info(f"Created unified paper with ID: {unified_id}")
        
        return UnifiedPaper(
            unified_id=unified_id,
            identifiers=identifiers,
            canonical_metadata=canonical_metadata,
            versions=versions,
            confidence_score=0.9  # High confidence for real data
        )
    
    def _normalize_to_canonical(self, paper_data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Normalize paper data from any source to canonical format"""
        
        # Extract basic fields
        title = paper_data.get('title', '')
        abstract = paper_data.get('abstract', '')
        authors = paper_data.get('authors', [])
        year = paper_data.get('year')
        venue = paper_data.get('venue', '')
        
        # Handle different author formats
        if isinstance(authors, str):
            authors = [author.strip() for author in authors.split(';') if author.strip()]
        elif isinstance(authors, list):
            authors = [str(author) for author in authors if author]
        
        # Extract year from date if needed
        if not year and 'published_date' in paper_data:
            try:
                year = int(paper_data['published_date'][:4])
            except (ValueError, TypeError):
                pass
        
        return {
            'title': title,
            'abstract': abstract,
            'authors': authors,
            'year': year,
            'venue': venue,
            'source': source,
            'identifier': paper_data.get('identifier', ''),
            'url': paper_data.get('url', ''),
            'citation_count': paper_data.get('citation_count', 0),
            'reference_count': paper_data.get('reference_count', 0)
        }
    
    async def _cache_mapping(self, unified_paper: UnifiedPaper):
        """Store unified mapping in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO paper_mappings 
            (unified_id, doi, arxiv_id, semantic_scholar_id, pmid, isbn,
             canonical_metadata, versions, citation_data, confidence_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            unified_paper.unified_id,
            unified_paper.identifiers.get('doi'),
            unified_paper.identifiers.get('arxiv'),
            unified_paper.identifiers.get('semantic_scholar'),
            unified_paper.identifiers.get('pmid'),
            unified_paper.identifiers.get('isbn'),
            json.dumps(unified_paper.canonical_metadata),
            json.dumps(unified_paper.versions),
            json.dumps(unified_paper.citation_data) if unified_paper.citation_data else None,
            unified_paper.confidence_score
        ))
        
        conn.commit()
        conn.close()