"""OpenCitations API client"""

from typing import Dict, List, Optional, Any
from arcane_mcp.data_sources.base_client import BaseAPIClient, RateLimitConfig

class OpenCitationsClient(BaseAPIClient):
    """OpenCitations API client for citation data"""
    
    def __init__(self, access_token: Optional[str] = None):
        super().__init__(
            base_url="https://opencitations.net/index/api/v2/",
            rate_limit_config=RateLimitConfig(requests_per_second=2.0),
        )
        self.access_token = access_token
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with access token if available"""
        headers = {'User-Agent': 'ARCANE/1.0'}
        if self.access_token:
            headers['authorization'] = self.access_token
        return headers
    
    async def search_papers(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """OpenCitations doesn't have search - return empty list"""
        return []
    
    async def get_paper(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Get paper metadata from OpenCitations"""
        if not (identifier.startswith('10.') or identifier.startswith('doi:')):
            return None
        
        clean_doi = identifier.replace('doi:', '')
        url = f"{self.base_url}metadata/doi:{clean_doi}"
        headers = self._get_headers()
        
        response = await self._make_request(url, headers=headers)
        if response and isinstance(response, list) and len(response) > 0:
            return self._normalize_oc_paper(response[0])
        
        return None
    
    async def get_citations(self, doi: str) -> Dict[str, Any]:
        """Get citation data for a DOI"""
        clean_doi = doi.replace('doi:', '')
        
        incoming_url = f"{self.base_url}citations/doi:{clean_doi}"
        outgoing_url = f"{self.base_url}references/doi:{clean_doi}"
        
        headers = self._get_headers()
        
        incoming_response = await self._make_request(incoming_url, headers=headers)
        outgoing_response = await self._make_request(outgoing_url, headers=headers)
        
        citation_data = {
            'doi': clean_doi,
            'incoming_citations': [],
            'outgoing_references': [],
            'citation_count': 0,
            'reference_count': 0
        }
        
        if incoming_response and isinstance(incoming_response, list):
            citation_data['incoming_citations'] = incoming_response
            citation_data['citation_count'] = len(incoming_response)
        
        if outgoing_response and isinstance(outgoing_response, list):
            citation_data['outgoing_references'] = outgoing_response
            citation_data['reference_count'] = len(outgoing_response)
        
        return citation_data
    
    def _normalize_oc_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize OpenCitations paper data"""
        return {
            'source': 'opencitations',
            'identifier': paper_data.get('id', ''),
            'title': paper_data.get('title', ''),
            'authors': paper_data.get('author', '').split('; ') if paper_data.get('author') else [],
            'year': paper_data.get('year'),
            'venue': paper_data.get('venue', ''),
            'doi': paper_data.get('doi', ''),
            'type': paper_data.get('type', '')
        }
    
    async def build_citation_graph(self, doi: str, depth: int = 2) -> Dict[str, Any]:
        """Build citation graph around a paper"""
        citation_graph = {
            'nodes': [],
            'edges': [],
            'center_doi': doi,
            'depth': depth
        }
        
        visited = set()
        to_process = [(doi, 0)]
        
        while to_process and len(citation_graph['nodes']) < 100:
            current_doi, current_depth = to_process.pop(0)
            
            if current_doi in visited or current_depth > depth:
                continue
            
            visited.add(current_doi)
            
            citation_data = await self.get_citations(current_doi)
            
            citation_graph['nodes'].append({
                'doi': current_doi,
                'depth': current_depth,
                'citation_count': citation_data['citation_count'],
                'reference_count': citation_data['reference_count']
            })
            
            if current_depth < depth:
                for citation in citation_data['incoming_citations'][:20]:
                    citing_doi = citation.get('citing')
                    if citing_doi and citing_doi not in visited:
                        citation_graph['edges'].append({
                            'source': citing_doi,
                            'target': current_doi,
                            'type': 'cites'
                        })
                        to_process.append((citing_doi, current_depth + 1))
                
                for reference in citation_data['outgoing_references'][:20]:
                    cited_doi = reference.get('cited')
                    if cited_doi and cited_doi not in visited:
                        citation_graph['edges'].append({
                            'source': current_doi,
                            'target': cited_doi,
                            'type': 'cites'
                        })
                        to_process.append((cited_doi, current_depth + 1))
        
        return citation_graph