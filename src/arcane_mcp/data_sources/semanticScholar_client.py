"""Semantic Scholar API client"""

from typing import Dict, List, Optional, Any
from arcane_mcp.data_sources.base_client import BaseAPIClient, RateLimitConfig

class SemanticScholarClient(BaseAPIClient):
    """Semantic Scholar API client"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            base_url="https://api.semanticscholar.org/graph/v1/",
            rate_limit_config=RateLimitConfig(requests_per_second=1.0 if not api_key else 10.0),
            api_key=api_key
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with API key if available"""
        headers = {'User-Agent': 'ARCANE/1.0'}
        if self.api_key:
            headers['x-api-key'] = self.api_key
        return headers
    
    async def search_papers(self, query: str, limit: int = 10, 
                          fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search papers using Semantic Scholar search API"""
        
        if fields is None:
            fields = [
                'paperId', 'title', 'abstract', 'authors', 'year', 'venue',
                'citationCount', 'referenceCount', 'externalIds', 'url'
            ]
        
        url = f"{self.base_url}paper/search"
        params = {
            'query': query,
            'limit': limit,
            'fields': ','.join(fields)
        }
        
        headers = self._get_headers()
        response = await self._make_request(url, params, headers)
        
        if response and 'data' in response:
            return [self._normalize_s2_paper(paper) for paper in response['data']]
        
        return []
    
    async def get_paper(self, identifier: str, 
                       fields: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Get paper by Semantic Scholar ID, DOI, or external ID"""
        
        if fields is None:
            fields = [
                'paperId', 'title', 'abstract', 'authors', 'year', 'venue',
                'citationCount', 'referenceCount', 'externalIds', 'url'
            ]
        
        # Determine URL based on identifier type
        if identifier.startswith('10.') or identifier.startswith('doi:'):
            clean_identifier = identifier.replace('doi:', '')
            url = f"{self.base_url}paper/DOI:{clean_identifier}"
        elif identifier.replace('.', '').replace('v', '').replace('-', '').isdigit():
            url = f"{self.base_url}paper/ARXIV:{identifier}"
        else:
            url = f"{self.base_url}paper/{identifier}"
        
        params = {'fields': ','.join(fields)}
        headers = self._get_headers()
        
        response = await self._make_request(url, params, headers)
        if response:
            return self._normalize_s2_paper(response)
        
        return None
    
    def _normalize_s2_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Semantic Scholar paper data"""
        
        authors = []
        if 'authors' in paper_data:
            for author in paper_data['authors']:
                if isinstance(author, dict) and 'name' in author:
                    authors.append(author['name'])
                elif isinstance(author, str):
                    authors.append(author)
        
        external_ids = paper_data.get('externalIds', {}) or {}
        
        return {
            'source': 'semantic_scholar',
            'identifier': paper_data.get('paperId'),
            'title': paper_data.get('title', ''),
            'abstract': paper_data.get('abstract', ''),
            'authors': authors,
            'year': paper_data.get('year'),
            'venue': paper_data.get('venue', ''),
            'citation_count': paper_data.get('citationCount', 0),
            'reference_count': paper_data.get('referenceCount', 0),
            'external_ids': external_ids,
            'url': paper_data.get('url')
        }
    
    async def get_paper_citations(self, paper_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get citations for a paper"""
        url = f"{self.base_url}paper/{paper_id}/citations"
        params = {
            'limit': limit,
            'fields': 'paperId,title,authors,year,venue,citationCount'
        }
        headers = self._get_headers()
        
        response = await self._make_request(url, params, headers)
        if response and 'data' in response:
            citations = []
            for item in response['data']:
                if 'citingPaper' in item:
                    citations.append(self._normalize_s2_paper(item['citingPaper']))
            return citations
        
        return []
    
    async def get_paper_references(self, paper_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get references for a paper"""
        url = f"{self.base_url}paper/{paper_id}/references"
        params = {
            'limit': limit,
            'fields': 'paperId,title,authors,year,venue,citationCount'
        }
        headers = self._get_headers()
        
        response = await self._make_request(url, params, headers)
        if response and 'data' in response:
            references = []
            for item in response['data']:
                if 'citedPaper' in item:
                    references.append(self._normalize_s2_paper(item['citedPaper']))
            return references
        
        return []