"""arXiv API client"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any
from arcane_mcp.data_sources.base_client import BaseAPIClient, RateLimitConfig
from ..config.config_manager import config_manager

class ArxivClient(BaseAPIClient):
    """arXiv API client using their REST API"""
    
    def __init__(self):
        # Get configuration values
        base_url = config_manager.get('api', 'arxiv.base_url', 'http://export.arxiv.org/api/')
        rate_limit = config_manager.get('api', 'arxiv.rate_limit', 1.0)
        
        super().__init__(
            base_url=base_url,
            rate_limit_config=RateLimitConfig(requests_per_second=rate_limit),
        )
    
    async def search_papers(self, query: str, max_results: int = 10, 
                          categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search arXiv papers"""
        
        # Build search query
        search_terms = []
        if query:
            search_terms.append(f'ti:"{query}" OR abs:"{query}"')
        
        if categories:
            cat_filter = " OR ".join([f"cat:{cat}" for cat in categories])
            search_terms.append(f"({cat_filter})")
        
        search_query = " AND ".join(search_terms) if search_terms else "all"
        
        url = f"{self.base_url}query"
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        response = await self._make_request(url, params)
        if not response:
            return []
        
        # Parse XML response
        try:
            root = ET.fromstring(response)
            entries = []
            
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                paper_data = self._parse_arxiv_entry(entry, ns)
                if paper_data:
                    entries.append(paper_data)
            
            return entries
            
        except ET.ParseError:
            return []
    
    def _parse_arxiv_entry(self, entry, namespaces) -> Dict[str, Any]:
        """Parse individual arXiv entry from XML"""
        try:
            title = entry.find('atom:title', namespaces)
            title_text = title.text.strip().replace('\n', ' ') if title is not None else ""
            
            summary = entry.find('atom:summary', namespaces)
            abstract = summary.text.strip().replace('\n', ' ') if summary is not None else ""
            
            id_elem = entry.find('atom:id', namespaces)
            arxiv_url = id_elem.text if id_elem is not None else ""
            arxiv_id = arxiv_url.split('/')[-1] if arxiv_url else ""
            
            authors = []
            for author in entry.findall('atom:author', namespaces):
                name_elem = author.find('atom:name', namespaces)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            
            published = entry.find('atom:published', namespaces)
            pub_date = published.text[:10] if published is not None else ""
            
            categories = []
            for category in entry.findall('atom:category', namespaces):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            return {
                'source': 'arxiv',
                'identifier': arxiv_id,
                'title': title_text,
                'abstract': abstract,
                'authors': authors,
                'published_date': pub_date,
                'categories': categories,
                'url': arxiv_url,
                'pdf_url': arxiv_url.replace('/abs/', '/pdf/') + '.pdf'
            }
            
        except Exception:
            return {}
    
    async def get_paper(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Get specific paper by arXiv ID"""
        clean_id = identifier.split('v')[0] if 'v' in identifier else identifier
        
        url = f"{self.base_url}query"
        params = {
            'id_list': clean_id,
            'max_results': 1
        }
        
        response = await self._make_request(url, params)
        if not response:
            return None
        
        try:
            root = ET.fromstring(response)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            entry = root.find('atom:entry', ns)
            if entry is not None:
                return self._parse_arxiv_entry(entry, ns)
            
        except ET.ParseError:
            pass
        
        return None