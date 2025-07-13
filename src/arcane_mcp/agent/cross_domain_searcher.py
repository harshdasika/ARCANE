"""
Cross-Domain Searcher - Discovers similar solutions across scientific domains
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import math
import weave
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .problem_abstractor import AbstractedProblem, MathematicalPattern, ProblemType, MathematicalStructure
from ..config.config_manager import config_manager

logger = logging.getLogger(__name__)

@dataclass
class SimilarityScore:
    """Represents similarity between problems"""
    mathematical_similarity: float
    methodological_similarity: float
    structural_similarity: float
    overall_similarity: float
    confidence: float

@dataclass
class CrossDomainMatch:
    """Represents a paper that matches the abstracted problem across domains"""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    domain: str
    year: Optional[int]
    citation_count: int
    similarity_score: SimilarityScore
    matching_patterns: List[str]
    relevance_explanation: str

@dataclass
class SearchResult:
    """Complete search results across domains"""
    original_problem: AbstractedProblem
    matches: List[CrossDomainMatch]
    search_strategy: Dict[str, Any]
    domains_searched: List[str]
    total_papers_analyzed: int
    search_confidence: float

class CrossDomainSearcher:
    """Searches for structurally similar problems across scientific domains"""
    
    def __init__(self, arcane_clients: Dict[str, Any]):
        """
        Initialize with ARCANE API clients
        
        Args:
            arcane_clients: Dictionary containing ARCANE API clients
        """
        self.logger = logging.getLogger(__name__)
        self.arxiv_client = arcane_clients.get('arxiv')
        self.semantic_scholar_client = arcane_clients.get('semantic_scholar')
        self.opencitations_client = arcane_clients.get('opencitations')
        self.identifier_resolver = arcane_clients.get('identifier_resolver')
        
        # Initialize semantic similarity model
        self._initialize_semantic_model()
        self._initialize_domain_mappings()
        self._initialize_similarity_weights()
        self._initialize_mathematical_patterns()
    
    def _initialize_semantic_model(self):
        """Initialize semantic similarity model for better cross-domain matching"""
        try:
            # Get model name from configuration
            model_name = config_manager.get('main', 'semantic_model.name', 'all-MiniLM-L6-v2')
            fallback_models = config_manager.get('main', 'semantic_model.fallback_models', [])
            
            # Try primary model first
            try:
                self.semantic_model = SentenceTransformer(model_name)
                self.logger.info(f"Semantic similarity model initialized: {model_name}")
            except Exception as e:
                self.logger.warning(f"Primary model {model_name} failed: {e}, trying fallbacks")
                
                # Try fallback models
                for fallback_model in fallback_models:
                    try:
                        self.semantic_model = SentenceTransformer(fallback_model)
                        self.logger.info(f"Using fallback semantic model: {fallback_model}")
                        break
                    except Exception as fe:
                        self.logger.warning(f"Fallback model {fallback_model} failed: {fe}")
                        continue
                else:
                    raise Exception("All models failed to load")
                    
        except Exception as e:
            self.logger.warning(f"Failed to initialize semantic model: {e}")
            self.semantic_model = None
    
    def _initialize_mathematical_patterns(self):
        """Initialize mathematical pattern recognition patterns from configuration"""
        
        # Load mathematical structures from domain config
        math_structures_config = config_manager.get('domain', 'mathematical_structures', {})
        
        # Core mathematical concepts that appear across domains (from config or defaults)
        concept_mappings = config_manager.get('domain', 'concept_mappings', {})
        
        # Ensure we have valid defaults and no None values
        self.mathematical_concepts = {
            "optimization": concept_mappings.get("optimization", [
                "minimize", "maximize", "optimal", "objective function", "constraint",
                "feasible region", "gradient", "convex", "linear programming"
            ]) or [
                "minimize", "maximize", "optimal", "objective function", "constraint",
                "feasible region", "gradient", "convex", "linear programming"
            ],
            "uncertainty_quantification": concept_mappings.get("uncertainty_quantification", [
                "uncertainty", "probability", "confidence interval", "bayesian",
                "monte carlo", "variance", "covariance", "error propagation"
            ]) or [
                "uncertainty", "probability", "confidence interval", "bayesian",
                "monte carlo", "variance", "covariance", "error propagation"
            ],
            "iterative_methods": concept_mappings.get("iterative_methods", [
                "iterative", "adaptive", "sequential", "feedback", "refinement",
                "convergence", "stability", "learning rate"
            ]) or [
                "iterative", "adaptive", "sequential", "feedback", "refinement",
                "convergence", "stability", "learning rate"
            ],
            "prediction_modeling": concept_mappings.get("prediction_modeling", [
                "prediction", "forecast", "model", "regression", "classification",
                "supervised learning", "unsupervised learning", "cross-validation"
            ]) or [
                "prediction", "forecast", "model", "regression", "classification",
                "supervised learning", "unsupervised learning", "cross-validation"
            ],
            "active_learning": concept_mappings.get("active_learning", [
                "active learning", "exploration", "exploitation", "acquisition function",
                "uncertainty sampling", "query strategy", "experimental design"
            ]) or [
                "active learning", "exploration", "exploitation", "acquisition function",
                "uncertainty sampling", "query strategy", "experimental design"
            ],
            "control_theory": concept_mappings.get("control_theory", [
                "control", "feedback", "stability", "robustness", "dynamics",
                "system identification", "state estimation", "kalman filter"
            ]) or [
                "control", "feedback", "stability", "robustness", "dynamics",
                "system identification", "state estimation", "kalman filter"
            ]
        }
        
        # Mathematical structure patterns (from config or defaults)
        structure_defaults = {
            MathematicalStructure.OPTIMIZATION_THEORY: [
                "optimization problem", "objective function", "constraint satisfaction",
                "feasibility", "pareto optimal", "multi-objective"
            ],
            MathematicalStructure.PROBABILITY_THEORY: [
                "probability distribution", "random variable", "stochastic process",
                "bayesian inference", "likelihood", "prior distribution"
            ],
            MathematicalStructure.GRAPH_THEORY: [
                "graph", "network", "node", "edge", "connectivity", "path",
                "topology", "clustering coefficient"
            ],
            MathematicalStructure.LINEAR_ALGEBRA: [
                "matrix", "vector", "eigenvalue", "decomposition", "linear system",
                "transformation", "basis", "subspace"
            ],
            MathematicalStructure.DIFFERENTIAL_EQUATIONS: [
                "differential equation", "ode", "pde", "boundary condition",
                "initial condition", "solution", "stability analysis"
            ]
        }
        
        self.structure_patterns = {}
        for structure, default_patterns in structure_defaults.items():
            config_key = structure.value.lower().replace(' ', '_') if hasattr(structure, 'value') else str(structure).lower()
            config_patterns = math_structures_config.get(config_key)
            self.structure_patterns[structure] = config_patterns if config_patterns is not None else default_patterns
    
    def _initialize_domain_mappings(self):
        """Initialize domain-specific search strategies from configuration"""
        
        # Load domain vocabularies from config
        domain_vocabularies = config_manager.get('domain', 'domain_vocabularies', {})
        
        # Build domain search strategies from config vocabularies
        self.domain_search_strategies = {}
        
        for domain_name, vocab in domain_vocabularies.items():
            # Map scientific domains to more specific domain keys
            search_domain = self._map_scientific_to_search_domain(domain_name)
            
            self.domain_search_strategies[search_domain] = {
                "primary_terms": (vocab.get('core_terms', []) or [])[:10],  # Limit to top 10
                "method_terms": (vocab.get('methods', []) or [])[:10],      # Limit to top 10
                "venues": self._get_domain_venues(domain_name)
            }
        
        # Add fallback domains if config is empty
        if not self.domain_search_strategies:
            self.domain_search_strategies = {
                "biomolecular": {
                    "primary_terms": ["molecular", "protein", "drug", "chemical", "biological"],
                    "method_terms": ["design", "discovery", "optimization", "prediction"],
                    "venues": ["Nature", "Science", "Cell", "PNAS", "Bioinformatics"]
                },
                "computer_science": {
                    "primary_terms": ["learning", "neural", "network", "algorithm", "model"],
                    "method_terms": ["training", "optimization", "prediction", "classification"],
                    "venues": ["ICML", "NeurIPS", "ICLR", "JMLR"]
                }
            }
        
        # Load mathematical pattern priorities (with defaults)
        self.domain_math_priorities = {
            "biomolecular": [MathematicalStructure.OPTIMIZATION_THEORY, MathematicalStructure.PROBABILITY_THEORY],
            "computer_science": [MathematicalStructure.PROBABILITY_THEORY, MathematicalStructure.OPTIMIZATION_THEORY],
            "physics": [MathematicalStructure.DIFFERENTIAL_EQUATIONS, MathematicalStructure.LINEAR_ALGEBRA],
            "chemistry": [MathematicalStructure.OPTIMIZATION_THEORY, MathematicalStructure.GRAPH_THEORY],
            "biology": [MathematicalStructure.GRAPH_THEORY, MathematicalStructure.PROBABILITY_THEORY],
            "mathematics": [MathematicalStructure.OPTIMIZATION_THEORY, MathematicalStructure.PROBABILITY_THEORY],
            "engineering": [MathematicalStructure.OPTIMIZATION_THEORY, MathematicalStructure.DIFFERENTIAL_EQUATIONS],
            "economics": [MathematicalStructure.OPTIMIZATION_THEORY, MathematicalStructure.PROBABILITY_THEORY]
        }
    
    def _map_scientific_to_search_domain(self, scientific_domain: str) -> str:
        """Map scientific domain names to search domain keys"""
        domain_mappings = {
            'biology': 'biomolecular',
            'chemistry': 'biomolecular', 
            'computer_science': 'machine_learning',
            'physics': 'physics',
            'mathematics': 'mathematics',
            'engineering': 'engineering',
            'economics': 'finance'
        }
        return domain_mappings.get(scientific_domain, scientific_domain)
    
    def _get_domain_venues(self, domain: str) -> List[str]:
        """Get high-quality venues for a domain"""
        venue_map = {
            'biology': ["Nature", "Science", "Cell", "PNAS", "Bioinformatics"],
            'chemistry': ["JACS", "Angewandte Chemie", "Nature Chemistry", "Science"],
            'computer_science': ["ICML", "NeurIPS", "ICLR", "JMLR", "AAAI"],
            'physics': ["Physical Review", "Nature Physics", "Science"],
            'mathematics': ["Annals of Mathematics", "Inventiones", "JAMS"],
            'engineering': ["ICRA", "IROS", "RSS", "IJRR"],
            'economics': ["Journal of Finance", "Quantitative Finance"]
        }
        return venue_map.get(domain, [])
    
    def _initialize_similarity_weights(self):
        """Initialize weights for similarity calculations"""
        self.similarity_weights = config_manager.get('similarity', 'weights', {
            "mathematical_structures": 0.25,
            "problem_types": 0.20,
            "semantic_similarity": 0.25,
            "mathematical_patterns": 0.15,
            "methodological_keywords": 0.10,
            "structural_complexity": 0.05
        })
    
    @weave.op()
    async def search_cross_domain(self, abstracted_problem: AbstractedProblem, 
                                max_results: int = 20) -> SearchResult:
        """
        Main method to search for similar solutions across domains
        
        Args:
            abstracted_problem: The abstracted research problem
            max_results: Maximum number of results to return
            
        Returns:
            SearchResult containing cross-domain matches
        """
        try:
            self.logger.info("Starting cross-domain search")
            
            # Generate search strategies
            search_strategies = await self._generate_search_strategies(abstracted_problem)
            
            # Execute searches across domains
            all_papers = await self._execute_multi_domain_search(search_strategies, max_results * 3)
            
            # Calculate similarity scores
            matches = await self._calculate_similarity_scores(abstracted_problem, all_papers)
            
            # Rank and filter results
            final_matches = self._rank_and_filter_matches(matches, max_results)
            
            # Calculate search confidence
            search_confidence = self._calculate_search_confidence(
                abstracted_problem, final_matches, len(all_papers)
            )
            
            search_result = SearchResult(
                original_problem=abstracted_problem,
                matches=final_matches,
                search_strategy=search_strategies,
                domains_searched=list(search_strategies.keys()),
                total_papers_analyzed=len(all_papers),
                search_confidence=search_confidence
            )
            
            self.logger.info(f"Cross-domain search completed: {len(final_matches)} matches found")
            return search_result
            
        except Exception as e:
            self.logger.error(f"Error in cross-domain search: {str(e)}")
            raise
    
    @weave.op()
    async def _generate_search_strategies(self, abstracted_problem: AbstractedProblem) -> Dict[str, Dict[str, Any]]:
        """Generate domain-specific search strategies based on mathematical patterns"""
        
        pattern = abstracted_problem.mathematical_pattern
        original_domain = abstracted_problem.domain_context.primary_domain
        
        strategies = {}
        
        # For each potential target domain
        for domain, domain_info in self.domain_search_strategies.items():
            # Skip the original domain (we want cross-domain solutions)
            if domain == original_domain:
                continue
            
            # Calculate domain relevance based on mathematical patterns
            relevance_score = self._calculate_domain_relevance(pattern, domain)
            
            if relevance_score > 0.1:  # Lower threshold for broader cross-domain search
                strategies[domain] = {
                    "primary_queries": self._generate_primary_queries(pattern, domain_info),
                    "mathematical_queries": self._generate_mathematical_queries(pattern),
                    "methodological_queries": self._generate_methodological_queries(pattern),
                    "relevance_score": relevance_score,
                    "max_results": max(5, int(relevance_score * 15))
                }
        
        return strategies
    
    def _calculate_domain_relevance(self, pattern: MathematicalPattern, domain: str) -> float:
        """Calculate how relevant a domain is for the given mathematical pattern"""
        
        # Start with base relevance for all domains to ensure cross-domain search
        relevance = 0.3  # Base relevance for broader cross-domain discovery
        
        if domain not in self.domain_math_priorities:
            return 0.5  # Default relevance for unknown domains
        
        domain_priorities = self.domain_math_priorities[domain]
        
        # Calculate relevance based on matching mathematical structures
        for structure in (pattern.mathematical_structures or []):
            if structure in domain_priorities:
                priority_index = domain_priorities.index(structure)
                # Higher score for higher priority structures
                relevance += 0.3 / (priority_index + 1)
        
        # Strong boost for specific problem types that are common across domains
        cross_domain_problems = [
            ProblemType.OPTIMIZATION, ProblemType.UNCERTAINTY_QUANTIFICATION,
            ProblemType.ACTIVE_LEARNING, ProblemType.PREDICTION, ProblemType.REINFORCEMENT_LEARNING
        ]
        
        for prob_type in (pattern.problem_types or []):
            if prob_type in cross_domain_problems:
                relevance += 0.25  # Increased boost for cross-domain problems
        
        # Special boost for uncertainty + iterative patterns (common in biomolecular design)
        if pattern.iterative_process and (pattern.uncertainty_sources or []):
            relevance += 0.2
        
        # Keywords that suggest cross-domain applicability
        cross_domain_keywords = [
            "uncertainty", "confidence", "adaptive", "iterative", "active learning",
            "optimization", "prediction", "policy", "feedback", "covariate shift"
        ]
        
        keyword_matches = sum(1 for keyword in cross_domain_keywords 
                            if any(keyword in term.lower() for term in (pattern.domain_agnostic_terms or [])))
        relevance += keyword_matches * 0.1
        
        return min(1.0, relevance)
    
    def _generate_primary_queries(self, pattern: MathematicalPattern, domain_info: Dict[str, Any]) -> List[str]:
        """Generate primary search queries for a domain"""
        
        queries = []
        domain_terms = domain_info.get("primary_terms", []) or []
        method_terms = domain_info.get("method_terms", []) or []
        
        # Combine domain terms with problem types
        problem_types = pattern.problem_types or []
        for prob_type in problem_types[:2]:  # Use top 2 problem types
            if prob_type != ProblemType.UNKNOWN:
                try:
                    prob_term = prob_type.value.replace('_', ' ')
                except AttributeError:
                    prob_term = str(prob_type).replace('_', ' ')
                for domain_term in domain_terms[:2]:
                    queries.append(f"{domain_term} {prob_term}")
        
        # Combine domain terms with mathematical structures
        math_structures = pattern.mathematical_structures or []
        for structure in math_structures[:2]:
            try:
                struct_term = structure.value.replace('_', ' ')
            except AttributeError:
                struct_term = str(structure).replace('_', ' ')
            for domain_term in domain_terms[:2]:
                queries.append(f"{domain_term} {struct_term}")
        
        # Combine domain and method terms
        for domain_term in domain_terms[:2]:
            for method_term in method_terms[:2]:
                queries.append(f"{domain_term} {method_term}")
        
        return queries[:6]  # Limit to 6 primary queries
    
    def _generate_mathematical_queries(self, pattern: MathematicalPattern) -> List[str]:
        """Generate queries focused on mathematical structures"""
        
        queries = []
        
        # Use domain-agnostic mathematical terms
        domain_agnostic_terms = pattern.domain_agnostic_terms or []
        for term in domain_agnostic_terms[:5]:
            queries.append(term)
        
        # Add mathematical concept queries based on pattern
        math_structures = pattern.mathematical_structures or []
        for structure in math_structures:
            if structure in self.structure_patterns:
                # Use mathematical pattern terms instead of raw structure names
                pattern_terms = self.structure_patterns[structure] or []
                queries.extend(pattern_terms[:2])  # Top 2 pattern terms per structure
        
        # Add concept-based queries
        for concept, terms in self.mathematical_concepts.items():
            # Check if this concept is relevant to the pattern
            if self._is_concept_relevant(concept, pattern):
                # Ensure terms is a valid list
                if terms is None:
                    self.logger.warning(f"Concept '{concept}' has None terms, skipping")
                    continue
                
                concept_terms = terms if isinstance(terms, list) else []
                if concept_terms:  # Only add if we have valid terms
                    queries.extend(concept_terms[:2])  # Top 2 terms per relevant concept
        
        # Combine mathematical structures with problem types using better terms
        for structure in math_structures:
            problem_types = pattern.problem_types or []
            for prob_type in problem_types:
                if prob_type != ProblemType.UNKNOWN:
                    # Use pattern terms instead of enum values
                    if structure in self.structure_patterns and self.structure_patterns[structure]:
                        struct_terms = self.structure_patterns[structure] or []
                        if struct_terms:
                            struct_term = struct_terms[0]  # Best pattern term
                            try:
                                prob_term = prob_type.value.replace('_', ' ')
                            except AttributeError:
                                prob_term = str(prob_type).replace('_', ' ')
                            queries.append(f"{struct_term} {prob_term}")
        
        return queries[:12]  # Increased limit for more comprehensive search
    
    def _is_concept_relevant(self, concept: str, pattern: MathematicalPattern) -> bool:
        """Check if a mathematical concept is relevant to the given pattern"""
        
        # Map concepts to problem types and structures
        concept_mappings = {
            "optimization": [ProblemType.OPTIMIZATION, MathematicalStructure.OPTIMIZATION_THEORY],
            "uncertainty_quantification": [ProblemType.UNCERTAINTY_QUANTIFICATION, 
                                         MathematicalStructure.PROBABILITY_THEORY],
            "iterative_methods": [],  # Check for iterative process flag
            "prediction_modeling": [ProblemType.PREDICTION, ProblemType.CLASSIFICATION],
            "active_learning": [ProblemType.ACTIVE_LEARNING],
            "control_theory": [ProblemType.REINFORCEMENT_LEARNING, MathematicalStructure.DIFFERENTIAL_EQUATIONS]
        }
        
        if concept not in concept_mappings:
            return False
        
        mapped_items = concept_mappings[concept]
        
        # Check problem types
        for item in mapped_items:
            if isinstance(item, ProblemType) and item in pattern.problem_types:
                return True
            elif hasattr(item, 'value') and item in pattern.mathematical_structures:
                return True
        
        # Special case for iterative methods
        if concept == "iterative_methods" and pattern.iterative_process:
            return True
            
        return False
    
    def _generate_methodological_queries(self, pattern: MathematicalPattern) -> List[str]:
        """Generate queries focused on methodological approaches"""
        
        queries = []
        
        # Use methodological keywords
        methodological_keywords = pattern.methodological_keywords or []
        for keyword in methodological_keywords[:5]:
            queries.append(keyword)
        
        # Add specific methodological patterns
        if pattern.iterative_process:
            queries.extend([
                "iterative optimization", "adaptive sampling", "sequential design",
                "active learning", "uncertainty-guided exploration", "feedback covariate shift"
            ])
        
        if pattern.uncertainty_sources:
            queries.extend([
                "uncertainty quantification", "robust optimization", "reliability analysis",
                "confidence estimation", "error propagation", "conformal prediction"
            ])
        
        # Add cross-domain methodological patterns
        problem_types = pattern.problem_types or []
        if ProblemType.PREDICTION in problem_types:
            queries.extend([
                "conformal prediction", "covariate shift", "adaptive experimental design",
                "uncertainty guided", "feedback optimization"
            ])
        
        # Specific pattern for uncertainty + prediction problems
        if (ProblemType.UNCERTAINTY_QUANTIFICATION in problem_types and 
            ProblemType.PREDICTION in problem_types):
            queries.extend([
                "conformal prediction feedback", "covariate shift biomolecular",
                "uncertainty guided design", "adaptive confidence",
                "conformal prediction", "biomolecular design uncertainty",
                "feedback covariate shift", "conformal prediction under feedback"
            ])
        
        # Enhanced patterns for active learning with uncertainty
        if (ProblemType.ACTIVE_LEARNING in problem_types and 
            pattern.uncertainty_sources):
            queries.extend([
                "active learning uncertainty", "sequential experimental design",
                "acquisition under covariate shift", "adaptive sampling uncertainty"
            ])
        
        # Patterns for iterative optimization with prediction
        if (pattern.iterative_process and 
            ProblemType.PREDICTION in problem_types):
            queries.extend([
                "iterative prediction refinement", "adaptive prediction models",
                "sequential prediction update", "online learning prediction"
            ])
        
        if ProblemType.REINFORCEMENT_LEARNING in problem_types:
            queries.extend([
                "policy evaluation", "feasibility analysis", "exploration strategy"
            ])
        
        return queries[:10]  # Increased limit for more comprehensive search
    
    @weave.op()
    async def _execute_multi_domain_search(self, search_strategies: Dict[str, Dict[str, Any]], 
                                         max_total_results: int) -> List[Dict[str, Any]]:
        """Execute searches across multiple domains using ARCANE clients"""
        
        all_papers = []
        
        # Collect all unique queries across domains
        all_queries = []
        for domain, strategy in search_strategies.items():
            queries = (strategy["primary_queries"] + 
                      strategy["mathematical_queries"] + 
                      strategy["methodological_queries"])
            all_queries.extend(queries)
        
        # Remove duplicates while preserving order
        unique_queries = list(dict.fromkeys(all_queries))
        
        # Execute searches in parallel
        search_tasks = []
        results_per_query = max(3, max_total_results // len(unique_queries))
        
        for query in unique_queries[:15]:  # Limit total queries
            # Search arXiv
            if self.arxiv_client:
                search_tasks.append(self._search_arxiv(query, results_per_query))
            
            # Search Semantic Scholar
            if self.semantic_scholar_client:
                search_tasks.append(self._search_semantic_scholar(query, results_per_query))
        
        # Execute all searches
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Aggregate results
        for result in search_results:
            if isinstance(result, list):
                all_papers.extend(result)
        
        # Remove duplicates based on title similarity
        unique_papers = self._remove_duplicate_papers(all_papers)
        
        return unique_papers[:max_total_results]
    
    async def _search_arxiv(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search arXiv using ARCANE client"""
        try:
            async with self.arxiv_client as client:
                papers = await client.search_papers(query=query, max_results=max_results)
                return papers or []
        except Exception as e:
            self.logger.warning(f"arXiv search failed for '{query}': {str(e)}")
            return []
    
    async def _search_semantic_scholar(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search Semantic Scholar using ARCANE client"""
        try:
            async with self.semantic_scholar_client as client:
                papers = await client.search_papers(query=query, limit=max_results)
                return papers or []
        except Exception as e:
            self.logger.warning(f"Semantic Scholar search failed for '{query}': {str(e)}")
            return []
    
    def _remove_duplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on title similarity"""
        
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            title = paper.get('title', '').lower().strip()
            if not title:
                continue
            
            # Simple deduplication based on title
            is_duplicate = False
            for seen_title in seen_titles:
                if self._calculate_title_similarity(title, seen_title) > 0.85:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.add(title)
        
        return unique_papers
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles"""
        # Simple word overlap similarity
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    @weave.op()
    async def _calculate_similarity_scores(self, abstracted_problem: AbstractedProblem, 
                                         papers: List[Dict[str, Any]]) -> List[CrossDomainMatch]:
        """Calculate similarity scores between the abstracted problem and found papers"""
        
        matches = []
        
        for paper in papers:
            similarity_score = await self._compute_paper_similarity(abstracted_problem, paper)
            
            # Use adaptive threshold based on pattern complexity
            adaptive_threshold = self._calculate_adaptive_threshold(abstracted_problem.mathematical_pattern)
            
            if similarity_score.overall_similarity > adaptive_threshold:
                match = CrossDomainMatch(
                    paper_id=paper.get('identifier', ''),
                    title=paper.get('title', ''),
                    authors=paper.get('authors', []),
                    abstract=paper.get('abstract', ''),
                    domain=self._infer_paper_domain(paper),
                    year=paper.get('year'),
                    citation_count=paper.get('citation_count', 0),
                    similarity_score=similarity_score,
                    matching_patterns=self._identify_matching_patterns(abstracted_problem, paper),
                    relevance_explanation=self._generate_relevance_explanation(
                        abstracted_problem, paper, similarity_score
                    )
                )
                matches.append(match)
        
        return matches
    
    @weave.op()
    async def _compute_paper_similarity(self, abstracted_problem: AbstractedProblem, 
                                      paper: Dict[str, Any]) -> SimilarityScore:
        """Compute detailed similarity score between problem and paper"""
        
        # Extract text content from paper
        paper_text = self._extract_paper_text(paper)
        
        # Create a mini-abstraction of the paper for comparison
        paper_pattern = await self._create_paper_pattern(paper_text)
        
        # Calculate component similarities
        math_similarity = self._calculate_mathematical_similarity(
            abstracted_problem.mathematical_pattern, paper_pattern
        )
        
        method_similarity = self._calculate_methodological_similarity(
            abstracted_problem.mathematical_pattern, paper_pattern
        )
        
        structural_similarity = self._calculate_structural_similarity(
            abstracted_problem.mathematical_pattern, paper_pattern
        )
        
        # Calculate semantic similarity
        semantic_similarity = await self._calculate_semantic_similarity(
            abstracted_problem, paper_text
        )
        
        # Calculate mathematical pattern similarity
        pattern_similarity = self._calculate_mathematical_pattern_similarity(
            abstracted_problem.mathematical_pattern, paper_pattern
        )
        
        # Calculate structural complexity similarity
        complexity_similarity = self._calculate_complexity_similarity(
            abstracted_problem.mathematical_pattern, paper_pattern
        )
        
        # Weighted overall similarity
        overall_similarity = (
            self.similarity_weights["mathematical_structures"] * math_similarity +
            self.similarity_weights["problem_types"] * method_similarity +
            self.similarity_weights["semantic_similarity"] * semantic_similarity +
            self.similarity_weights["mathematical_patterns"] * pattern_similarity +
            self.similarity_weights["methodological_keywords"] * structural_similarity +
            self.similarity_weights["structural_complexity"] * complexity_similarity
        )
        
        # Calculate confidence based on text quality and length
        confidence = self._calculate_similarity_confidence(paper_text, paper_pattern)
        
        return SimilarityScore(
            mathematical_similarity=math_similarity,
            methodological_similarity=method_similarity,
            structural_similarity=structural_similarity,
            overall_similarity=overall_similarity,
            confidence=confidence
        )
    
    def _extract_paper_text(self, paper: Dict[str, Any]) -> str:
        """Extract meaningful text from paper for analysis"""
        text_parts = []
        
        if paper.get('title'):
            text_parts.append(paper['title'])
        
        if paper.get('abstract'):
            text_parts.append(paper['abstract'])
        
        # Add other text fields if available
        for field in ['summary', 'description', 'keywords']:
            if paper.get(field):
                text_parts.append(str(paper[field]))
        
        return ' '.join(text_parts).lower()
    
    async def _create_paper_pattern(self, paper_text: str) -> MathematicalPattern:
        """Create a simplified mathematical pattern from paper text"""
        
        # This is a simplified version of the full problem abstractor
        # We only extract the most important patterns for similarity comparison
        
        problem_types = []
        math_structures = []
        methodological_keywords = []
        
        # Simple keyword matching for pattern detection
        from .problem_abstractor import ProblemAbstractor
        abstractor = ProblemAbstractor()
        
        # Detect problem types
        for prob_type, keywords in abstractor.problem_keywords.items():
            if any(keyword in paper_text for keyword in keywords):
                problem_types.append(prob_type)
        
        # Detect mathematical structures
        for structure, keywords in abstractor.math_keywords.items():
            if any(keyword in paper_text for keyword in keywords):
                math_structures.append(structure)
        
        # Extract methodological keywords
        method_keywords = [
            "machine learning", "optimization", "uncertainty", "active learning",
            "bayesian", "neural network", "regression", "classification"
        ]
        methodological_keywords = [kw for kw in method_keywords if kw in paper_text]
        
        return MathematicalPattern(
            problem_types=problem_types,
            mathematical_structures=math_structures,
            objective_function=None,
            constraints=[],
            variables=[],
            uncertainty_sources=[],
            iterative_process="iterative" in paper_text or "adaptive" in paper_text,
            domain_agnostic_terms=[],
            methodological_keywords=methodological_keywords,
            confidence_score=0.7  # Default confidence for paper patterns
        )
    
    def _calculate_mathematical_similarity(self, pattern1: MathematicalPattern, 
                                         pattern2: MathematicalPattern) -> float:
        """Calculate similarity based on mathematical structures"""
        
        if not pattern1.mathematical_structures or not pattern2.mathematical_structures:
            return 0.0
        
        set1 = set(pattern1.mathematical_structures)
        set2 = set(pattern2.mathematical_structures)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_methodological_similarity(self, pattern1: MathematicalPattern,
                                           pattern2: MathematicalPattern) -> float:
        """Calculate similarity based on problem types and methods"""
        
        # Problem type similarity
        type_similarity = 0.0
        if pattern1.problem_types and pattern2.problem_types:
            set1 = set(pattern1.problem_types)
            set2 = set(pattern2.problem_types)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            type_similarity = intersection / union if union > 0 else 0.0
        
        # Methodological keyword similarity
        keyword_similarity = 0.0
        if pattern1.methodological_keywords and pattern2.methodological_keywords:
            set1 = set(pattern1.methodological_keywords)
            set2 = set(pattern2.methodological_keywords)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            keyword_similarity = intersection / union if union > 0 else 0.0
        
        return (type_similarity + keyword_similarity) / 2
    
    def _calculate_structural_similarity(self, pattern1: MathematicalPattern,
                                       pattern2: MathematicalPattern) -> float:
        """Calculate similarity based on structural aspects"""
        
        similarity = 0.0
        
        # Iterative process similarity
        if pattern1.iterative_process == pattern2.iterative_process:
            similarity += 0.5
        
        # Uncertainty handling similarity
        if pattern1.uncertainty_sources and pattern2.uncertainty_sources:
            similarity += 0.3
        elif not pattern1.uncertainty_sources and not pattern2.uncertainty_sources:
            similarity += 0.1
        
        # Complexity similarity (based on number of components)
        complexity1 = len(pattern1.mathematical_structures) + len(pattern1.problem_types)
        complexity2 = len(pattern2.mathematical_structures) + len(pattern2.problem_types)
        
        if complexity1 > 0 and complexity2 > 0:
            complexity_similarity = 1 - abs(complexity1 - complexity2) / max(complexity1, complexity2)
            similarity += 0.2 * complexity_similarity
        
        return min(1.0, similarity)
    
    async def _calculate_semantic_similarity(self, abstracted_problem: AbstractedProblem, 
                                           paper_text: str) -> float:
        """Calculate semantic similarity between abstracted problem and paper text"""
        
        if not self.semantic_model:
            return 0.5  # Fallback similarity
        
        try:
            # Create problem description
            problem_description = self._create_problem_description(abstracted_problem)
            
            # Encode both texts
            problem_embedding = self.semantic_model.encode([problem_description[:512]])  # Limit text length
            paper_embedding = self.semantic_model.encode([paper_text[:512]])  # Limit text length
            
            # Calculate cosine similarity
            similarity = cosine_similarity(problem_embedding, paper_embedding)[0][0]
            
            return float(similarity)
            
        except Exception as e:
            self.logger.warning(f"Error in semantic similarity calculation: {e}")
            return 0.5  # Fallback similarity
    
    def _create_problem_description(self, abstracted_problem: AbstractedProblem) -> str:
        """Create a comprehensive problem description for similarity matching"""
        
        pattern = abstracted_problem.mathematical_pattern
        domain = abstracted_problem.domain_context.primary_domain
        
        # Build description components
        components = []
        
        # Problem types
        problem_types = pattern.problem_types or []
        if problem_types:
            type_names = []
            for pt in problem_types:
                if pt != ProblemType.UNKNOWN:
                    try:
                        type_names.append(pt.value.replace('_', ' '))
                    except AttributeError:
                        type_names.append(str(pt).replace('_', ' '))
            if type_names:
                components.append(f"Problem types: {', '.join(type_names)}")
        
        # Mathematical structures
        math_structures = pattern.mathematical_structures or []
        if math_structures:
            struct_names = []
            for ms in math_structures:
                try:
                    struct_names.append(ms.value.replace('_', ' '))
                except AttributeError:
                    struct_names.append(str(ms).replace('_', ' '))
            if struct_names:
                components.append(f"Mathematical structures: {', '.join(struct_names)}")
        
        # Domain-agnostic terms
        domain_agnostic_terms = pattern.domain_agnostic_terms or []
        if domain_agnostic_terms:
            components.append(f"Key concepts: {', '.join(domain_agnostic_terms[:5])}")
        
        # Methodological keywords
        methodological_keywords = pattern.methodological_keywords or []
        if methodological_keywords:
            components.append(f"Methods: {', '.join(methodological_keywords[:5])}")
        
        # Domain context
        if domain != "unknown":
            components.append(f"Domain: {domain}")
        
        # Special characteristics
        if pattern.iterative_process:
            components.append("Iterative process")
        
        if pattern.uncertainty_sources:
            components.append("Uncertainty quantification")
        
        return ". ".join(components)
    
    def _calculate_mathematical_pattern_similarity(self, pattern1: MathematicalPattern,
                                                 pattern2: MathematicalPattern) -> float:
        """Calculate similarity based on mathematical patterns and concepts"""
        
        similarity = 0.0
        
        # Check for mathematical concept overlap
        pattern1_concepts = self._extract_mathematical_concepts(pattern1)
        pattern2_concepts = self._extract_mathematical_concepts(pattern2)
        
        if pattern1_concepts and pattern2_concepts:
            concept_overlap = len(pattern1_concepts.intersection(pattern2_concepts))
            concept_union = len(pattern1_concepts.union(pattern2_concepts))
            similarity += 0.5 * (concept_overlap / concept_union if concept_union > 0 else 0)
        
        # Check for structural pattern similarity
        if pattern1.iterative_process and pattern2.iterative_process:
            similarity += 0.3
        
        if pattern1.uncertainty_sources and pattern2.uncertainty_sources:
            similarity += 0.2
        
        return min(1.0, similarity)
    
    def _extract_mathematical_concepts(self, pattern: MathematicalPattern) -> set:
        """Extract mathematical concepts present in a pattern"""
        
        concepts = set()
        
        # Map structures to concepts
        structure_to_concept = {
            MathematicalStructure.OPTIMIZATION_THEORY: "optimization",
            MathematicalStructure.PROBABILITY_THEORY: "uncertainty_quantification",
            MathematicalStructure.DIFFERENTIAL_EQUATIONS: "control_theory"
        }
        
        for structure in (pattern.mathematical_structures or []):
            if structure in structure_to_concept:
                concepts.add(structure_to_concept[structure])
        
        # Map problem types to concepts
        type_to_concept = {
            ProblemType.OPTIMIZATION: "optimization",
            ProblemType.UNCERTAINTY_QUANTIFICATION: "uncertainty_quantification",
            ProblemType.PREDICTION: "prediction_modeling",
            ProblemType.ACTIVE_LEARNING: "active_learning",
            ProblemType.REINFORCEMENT_LEARNING: "control_theory"
        }
        
        for prob_type in (pattern.problem_types or []):
            if prob_type in type_to_concept:
                concepts.add(type_to_concept[prob_type])
        
        # Check for iterative methods
        if pattern.iterative_process:
            concepts.add("iterative_methods")
        
        return concepts
    
    def _calculate_complexity_similarity(self, pattern1: MathematicalPattern,
                                       pattern2: MathematicalPattern) -> float:
        """Calculate similarity based on problem complexity"""
        
        # Calculate complexity scores
        complexity1 = (len(pattern1.mathematical_structures) + 
                      len(pattern1.problem_types) + 
                      (1 if pattern1.iterative_process else 0) +
                      (1 if pattern1.uncertainty_sources else 0))
        
        complexity2 = (len(pattern2.mathematical_structures) + 
                      len(pattern2.problem_types) + 
                      (1 if pattern2.iterative_process else 0) +
                      (1 if pattern2.uncertainty_sources else 0))
        
        if complexity1 == 0 and complexity2 == 0:
            return 1.0
        
        if complexity1 == 0 or complexity2 == 0:
            return 0.0
        
        # Calculate relative complexity similarity
        ratio = min(complexity1, complexity2) / max(complexity1, complexity2)
        return ratio
    
    def _calculate_adaptive_threshold(self, pattern: MathematicalPattern) -> float:
        """Calculate adaptive similarity threshold based on pattern complexity"""
        
        # Get configuration values
        base_threshold = config_manager.get('similarity', 'thresholds.base_threshold', 0.3)
        adjustments = config_manager.get('similarity', 'thresholds.adaptive_adjustments', {})
        bounds = config_manager.get('similarity', 'thresholds.bounds', {})
        
        # Calculate pattern complexity
        complexity = (len(pattern.mathematical_structures) + 
                     len(pattern.problem_types) + 
                     len(pattern.methodological_keywords))
        
        threshold = base_threshold
        
        # Lower threshold for more complex patterns (they're rarer)
        if complexity > 5:
            threshold -= adjustments.get('complex_pattern_reduction', 0.1)
        elif complexity > 3:
            threshold -= adjustments.get('moderate_pattern_reduction', 0.05)
        
        # Further adjust based on specific characteristics
        if pattern.iterative_process and (pattern.uncertainty_sources or []):
            threshold -= adjustments.get('uncertainty_iterative_boost', 0.05)
        
        if len(pattern.domain_agnostic_terms) > 7:
            threshold -= adjustments.get('rich_abstraction_boost', 0.03)
        
        # Ensure threshold stays within bounds
        min_threshold = bounds.get('min_threshold', 0.15)
        max_threshold = bounds.get('max_threshold', 0.45)
        return max(min_threshold, min(max_threshold, threshold))
    
    def _calculate_similarity_confidence(self, paper_text: str, pattern: MathematicalPattern) -> float:
        """Calculate confidence in the similarity score"""
        
        # Text quality indicators
        text_length = len(paper_text.split())
        has_abstract = "abstract" in paper_text.lower()
        technical_terms = sum(1 for term in ["algorithm", "method", "approach", "model"] 
                            if term in paper_text)
        
        confidence = 0.3  # Base confidence
        
        if text_length > 100:
            confidence += 0.2
        if has_abstract:
            confidence += 0.2
        if technical_terms > 2:
            confidence += 0.2
        if pattern.mathematical_structures:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _infer_paper_domain(self, paper: Dict[str, Any]) -> str:
        """Infer the scientific domain of a paper"""
        
        text = self._extract_paper_text(paper)
        
        domain_scores = {}
        for domain, strategy in self.domain_search_strategies.items():
            score = sum(1 for term in strategy["primary_terms"] if term in text)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return "unknown"
    
    def _identify_matching_patterns(self, abstracted_problem: AbstractedProblem, 
                                  paper: Dict[str, Any]) -> List[str]:
        """Identify specific patterns that match between problem and paper"""
        
        matches = []
        pattern = abstracted_problem.mathematical_pattern
        paper_text = self._extract_paper_text(paper)
        
        # Check for mathematical structure matches
        for structure in (pattern.mathematical_structures or []):
            struct_terms = structure.value.replace('_', ' ')
            if any(term in paper_text for term in struct_terms.split()):
                matches.append(f"Mathematical: {struct_terms}")
        
        # Check for problem type matches
        for prob_type in (pattern.problem_types or []):
            if prob_type != ProblemType.UNKNOWN:
                type_term = prob_type.value.replace('_', ' ')
                if type_term in paper_text:
                    matches.append(f"Problem Type: {type_term}")
        
        # Check for methodological matches
        for keyword in pattern.methodological_keywords:
            if keyword in paper_text:
                matches.append(f"Method: {keyword}")
        
        return matches
    
    def _generate_relevance_explanation(self, abstracted_problem: AbstractedProblem,
                                      paper: Dict[str, Any], 
                                      similarity: SimilarityScore) -> str:
        """Generate human-readable explanation of why this paper is relevant"""
        
        explanations = []
        
        if similarity.mathematical_similarity > 0.5:
            explanations.append("shares similar mathematical foundations")
        
        if similarity.methodological_similarity > 0.5:
            explanations.append("uses comparable methodological approaches")
        
        if similarity.structural_similarity > 0.5:
            explanations.append("has analogous problem structure")
        
        domain = self._infer_paper_domain(paper)
        original_domain = abstracted_problem.domain_context.primary_domain
        
        if domain != original_domain and domain != "unknown":
            explanations.append(f"demonstrates cross-domain applicability from {domain}")
        
        if not explanations:
            explanations.append("shows potential methodological overlap")
        
        return "This paper " + " and ".join(explanations) + "."
    
    def _rank_and_filter_matches(self, matches: List[CrossDomainMatch], 
                               max_results: int) -> List[CrossDomainMatch]:
        """Rank matches by overall similarity and filter to top results"""
        
        # Sort by overall similarity score (descending)
        ranked_matches = sorted(matches, 
                              key=lambda m: m.similarity_score.overall_similarity, 
                              reverse=True)
        
        # Apply additional filtering criteria
        filtered_matches = []
        
        for match in ranked_matches:
            # Minimum similarity threshold
            if match.similarity_score.overall_similarity < 0.3:
                continue
            
            # Minimum confidence threshold  
            if match.similarity_score.confidence < 0.4:
                continue
            
            # Avoid very old papers unless highly relevant
            if match.year and match.year < 2000 and match.similarity_score.overall_similarity < 0.7:
                continue
            
            filtered_matches.append(match)
            
            if len(filtered_matches) >= max_results:
                break
        
        return filtered_matches
    
    def _calculate_search_confidence(self, abstracted_problem: AbstractedProblem,
                                   matches: List[CrossDomainMatch], 
                                   total_analyzed: int) -> float:
        """Calculate overall confidence in the search results"""
        
        if not matches:
            return 0.1
        
        # Base confidence from problem abstraction
        base_confidence = abstracted_problem.abstraction_confidence
        
        # Coverage confidence (how many papers were analyzed)
        coverage_confidence = min(1.0, total_analyzed / 50)  # Optimal around 50+ papers
        
        # Result quality confidence
        avg_similarity = sum(m.similarity_score.overall_similarity for m in matches) / len(matches)
        quality_confidence = avg_similarity
        
        # Diversity confidence (multiple domains represented)
        unique_domains = len(set(m.domain for m in matches))
        diversity_confidence = min(1.0, unique_domains / 3)  # Optimal around 3+ domains
        
        # Validation confidence (known cross-domain cases)
        validation_confidence = self._validate_known_cross_domain_cases(matches, abstracted_problem)
        
        # Weighted average using configurable weights
        confidence_weights = config_manager.get('validation', 'confidence_weights', {
            'base_confidence': 0.25,
            'coverage_confidence': 0.15,
            'quality_confidence': 0.25,
            'diversity_confidence': 0.15,
            'validation_confidence': 0.20
        })
        
        overall_confidence = (
            confidence_weights.get('base_confidence', 0.25) * base_confidence +
            confidence_weights.get('coverage_confidence', 0.15) * coverage_confidence +
            confidence_weights.get('quality_confidence', 0.25) * quality_confidence +
            confidence_weights.get('diversity_confidence', 0.15) * diversity_confidence +
            confidence_weights.get('validation_confidence', 0.20) * validation_confidence
        )
        
        return min(1.0, overall_confidence)
    
    def _validate_known_cross_domain_cases(self, matches: List[CrossDomainMatch], 
                                         abstracted_problem: AbstractedProblem) -> float:
        """Validate against known successful cross-domain transfers using configuration"""
        
        validation_score = 0.0
        pattern = abstracted_problem.mathematical_pattern
        
        # Load known cases from configuration
        known_cases = config_manager.get('validation', 'known_cases', [])
        
        for case in known_cases:
            case_name = case.get('name', '')
            conditions = case.get('conditions', {})
            indicators = case.get('indicators', {})
            validation_weight = case.get('validation_weight', 0.1)
            required_matches = case.get('required_matches', {})
            
            # Check if this case applies to the current problem
            if self._case_applies_to_problem(pattern, conditions):
                
                # Check matches for this validation case
                for match in matches:
                    title_lower = (match.title or "").lower()
                    abstract_lower = (match.abstract or "").lower()
                    text = title_lower + " " + abstract_lower
                    
                    # Count indicator matches for each category
                    category_scores = {}
                    for category, terms in indicators.items():
                        category_scores[category] = sum(1 for term in terms if term in text)
                    
                    # Check if required matches are met
                    case_matched = True
                    for category, required_count in required_matches.items():
                        if category_scores.get(category, 0) < required_count:
                            case_matched = False
                            break
                    
                    if case_matched:
                        validation_score += validation_weight
                        self.logger.info(f"Validated known case: {case_name}")
                        break
        
        return min(1.0, validation_score)
    
    def _case_applies_to_problem(self, pattern: MathematicalPattern, conditions: Dict[str, Any]) -> bool:
        """Check if a validation case applies to the current problem pattern"""
        
        # Check problem types
        required_types = conditions.get('problem_types', [])
        if required_types:
            pattern_type_names = [pt.value if hasattr(pt, 'value') else str(pt).lower() 
                                for pt in (pattern.problem_types or [])]
            if not any(req_type in pattern_type_names for req_type in required_types):
                return False
        
        # Check iterative process requirement
        if 'iterative_process' in conditions:
            if pattern.iterative_process != conditions['iterative_process']:
                return False
        
        return True