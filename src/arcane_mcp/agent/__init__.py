"""
Cross-Domain Research Discovery Agent

This module provides AI agent capabilities for discovering research solutions
across scientific domains by abstracting mathematical patterns and finding
structurally similar problems in different fields.
"""

from .orchestrator import CrossDomainResearchAgent
from .problem_abstractor import ProblemAbstractor
from .cross_domain_searcher import CrossDomainSearcher
from .solution_translator import SolutionTranslator
from .reasoning_tracer import ReasoningTracer

__all__ = [
    "CrossDomainResearchAgent",
    "ProblemAbstractor", 
    "CrossDomainSearcher",
    "SolutionTranslator",
    "ReasoningTracer"
]