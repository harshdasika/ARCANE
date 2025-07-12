"""Base API client with rate limiting"""

import asyncio
import aiohttp
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class RateLimitConfig:
    requests_per_second: float
    burst_limit: int = 10

class RateLimiter:
    """Simple rate limiter for API requests"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests = []
    
    async def acquire(self):
        """Wait if necessary to respect rate limits"""
        now = time.time()
        
        # Remove old requests outside the window
        cutoff_time = now - 1.0
        self.requests = [req_time for req_time in self.requests if req_time > cutoff_time]
        
        # Check if we need to wait
        if len(self.requests) >= int(self.config.requests_per_second):
            sleep_time = 1.0 / self.config.requests_per_second
            await asyncio.sleep(sleep_time)
        
        self.requests.append(now)

class BaseAPIClient(ABC):
    """Abstract base class for all API clients"""
    
    def __init__(self, base_url: str, rate_limit_config: RateLimitConfig, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.rate_limiter = RateLimiter(rate_limit_config)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def search_papers(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search for papers"""
        pass
    
    @abstractmethod
    async def get_paper(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Get paper by identifier"""
        pass
    
    async def _make_request(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Optional[Any]:
        """Make rate-limited HTTP request"""
        await self.rate_limiter.acquire()
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    if 'application/json' in response.content_type:
                        return await response.json()
                    else:
                        return await response.text()
                elif response.status == 429:
                    await asyncio.sleep(60)
                    return await self._make_request(url, params, headers)
                else:
                    return None
        except Exception:
            return None