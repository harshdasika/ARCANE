"""
Problem Abstractor - Extracts mathematical and methodological patterns from research problems
"""

import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import weave

logger = logging.getLogger(__name__)

class ProblemType(Enum):
    OPTIMIZATION = "optimization"
    PREDICTION = "prediction"
    CLASSIFICATION = "classification"
    UNCERTAINTY_QUANTIFICATION = "uncertainty_quantification"
    ACTIVE_LEARNING = "active_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    STATISTICAL_INFERENCE = "statistical_inference"
    CONTROL_THEORY = "control_theory"
    UNKNOWN = "unknown"

class MathematicalStructure(Enum):
    LINEAR_ALGEBRA = "linear_algebra"
    PROBABILITY_THEORY = "probability_theory"
    GRAPH_THEORY = "graph_theory"
    DIFFERENTIAL_EQUATIONS = "differential_equations"
    OPTIMIZATION_THEORY = "optimization_theory"
    INFORMATION_THEORY = "information_theory"
    GAME_THEORY = "game_theory"
    TOPOLOGY = "topology"

@dataclass
class MathematicalPattern:
    """Represents the abstracted mathematical structure of a research problem"""
    problem_types: List[ProblemType]
    mathematical_structures: List[MathematicalStructure]
    objective_function: Optional[str]
    constraints: List[str]
    variables: List[str]
    uncertainty_sources: List[str]
    iterative_process: bool
    domain_agnostic_terms: List[str]
    methodological_keywords: List[str]
    confidence_score: float

@dataclass
class DomainContext:
    """Represents the domain-specific context of the problem"""
    primary_domain: str
    subdomain: Optional[str]
    domain_specific_terms: List[str]
    entities: List[str]
    processes: List[str]

@dataclass
class AbstractedProblem:
    """Complete abstraction of a research problem"""
    original_description: str
    mathematical_pattern: MathematicalPattern
    domain_context: DomainContext
    abstraction_confidence: float

class ProblemAbstractor:
    """Extracts mathematical and methodological patterns from research problem descriptions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_pattern_libraries()
    
    def _initialize_pattern_libraries(self):
        """Initialize pattern recognition libraries"""
        
        # Mathematical keywords for pattern recognition
        self.math_keywords = {
            MathematicalStructure.OPTIMIZATION_THEORY: [
                "minimize", "maximize", "optimal", "objective function", "cost function",
                "gradient", "convex", "global minimum", "local optimum", "constraint optimization"
            ],
            MathematicalStructure.PROBABILITY_THEORY: [
                "probability", "distribution", "bayesian", "prior", "posterior", "likelihood",
                "uncertainty", "confidence interval", "statistical", "stochastic", "random"
            ],
            MathematicalStructure.LINEAR_ALGEBRA: [
                "matrix", "vector", "eigenvalue", "decomposition", "linear system",
                "basis", "dimension", "subspace", "projection", "transformation"
            ],
            MathematicalStructure.GRAPH_THEORY: [
                "network", "graph", "node", "edge", "connectivity", "path",
                "topology", "clustering", "centrality", "adjacency"
            ],
            MathematicalStructure.DIFFERENTIAL_EQUATIONS: [
                "differential", "derivative", "integration", "dynamics", "evolution",
                "rate", "change", "temporal", "continuous", "flow"
            ],
            MathematicalStructure.INFORMATION_THEORY: [
                "entropy", "information", "mutual information", "compression",
                "encoding", "channel", "capacity", "redundancy"
            ]
        }
        
        # Problem type keywords
        self.problem_keywords = {
            ProblemType.OPTIMIZATION: [
                "optimize", "minimize", "maximize", "best", "optimal", "improve",
                "tune", "search", "find minimum", "find maximum"
            ],
            ProblemType.PREDICTION: [
                "predict", "forecast", "estimate", "model", "regression",
                "time series", "future", "anticipate"
            ],
            ProblemType.CLASSIFICATION: [
                "classify", "categorize", "label", "discriminate", "identify",
                "recognize", "detect", "distinguish"
            ],
            ProblemType.UNCERTAINTY_QUANTIFICATION: [
                "uncertainty", "confidence", "reliability", "robustness",
                "error bars", "variance", "risk", "sensitivity"
            ],
            ProblemType.ACTIVE_LEARNING: [
                "active learning", "iterative", "adaptive", "query", "select data",
                "information gain", "uncertainty sampling", "explore"
            ],
            ProblemType.REINFORCEMENT_LEARNING: [
                "policy", "reward", "agent", "environment", "action", "state",
                "reinforcement", "learning", "explore", "exploit"
            ]
        }
        
        # Domain recognition patterns
        self.domain_patterns = {
            "robotics": ["robot", "robotic", "manipulation", "navigation", "motion planning", "trajectory"],
            "biomolecular": ["molecular", "protein", "drug", "chemical", "biological", "biomolecular"],
            "computer_vision": ["image", "vision", "visual", "recognition", "detection", "segmentation"],
            "natural_language": ["language", "text", "linguistic", "nlp", "semantic", "syntax"],
            "finance": ["financial", "market", "trading", "portfolio", "risk", "investment"],
            "physics": ["physical", "quantum", "particle", "field", "energy", "force"],
            "materials": ["material", "crystal", "alloy", "composite", "structure", "properties"],
            "climate": ["climate", "weather", "atmospheric", "environmental", "temperature", "precipitation"],
            "neuroscience": ["neural", "brain", "neuron", "cognitive", "behavioral", "neurological"]
        }
    
    @weave.op()
    async def abstract_problem(self, problem_description: str) -> AbstractedProblem:
        """
        Main method to abstract a research problem into mathematical patterns
        
        Args:
            problem_description: Natural language description of the research problem
            
        Returns:
            AbstractedProblem containing mathematical patterns and domain context
        """
        try:
            self.logger.info(f"Abstracting problem: {problem_description[:100]}...")
            
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(problem_description)
            
            # Extract mathematical patterns
            mathematical_pattern = await self._extract_mathematical_patterns(cleaned_text)
            
            # Extract domain context
            domain_context = await self._extract_domain_context(cleaned_text)
            
            # Calculate overall confidence
            abstraction_confidence = self._calculate_abstraction_confidence(
                mathematical_pattern, domain_context
            )
            
            abstracted_problem = AbstractedProblem(
                original_description=problem_description,
                mathematical_pattern=mathematical_pattern,
                domain_context=domain_context,
                abstraction_confidence=abstraction_confidence
            )
            
            self.logger.info(f"Abstraction completed with confidence: {abstraction_confidence:.2f}")
            return abstracted_problem
            
        except Exception as e:
            self.logger.error(f"Error in problem abstraction: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for pattern extraction"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep mathematical notation
        text = re.sub(r'[^\w\s\+\-\*\/\=\(\)\[\]\{\}\<\>\.]', ' ', text)
        
        return text.strip()
    
    @weave.op()
    async def _extract_mathematical_patterns(self, text: str) -> MathematicalPattern:
        """Extract mathematical structures and problem types from text"""
        
        # Detect problem types
        problem_types = []
        for prob_type, keywords in self.problem_keywords.items():
            if any(keyword in text for keyword in keywords):
                problem_types.append(prob_type)
        
        # Detect mathematical structures
        math_structures = []
        for structure, keywords in self.math_keywords.items():
            if any(keyword in text for keyword in keywords):
                math_structures.append(structure)
        
        # Extract key components
        objective_function = self._extract_objective_function(text)
        constraints = self._extract_constraints(text)
        variables = self._extract_variables(text)
        uncertainty_sources = self._extract_uncertainty_sources(text)
        iterative_process = self._detect_iterative_process(text)
        
        # Generate domain-agnostic terms
        domain_agnostic_terms = self._generate_domain_agnostic_terms(
            problem_types, math_structures, text
        )
        
        # Extract methodological keywords
        methodological_keywords = self._extract_methodological_keywords(text)
        
        # Calculate pattern confidence
        confidence = self._calculate_pattern_confidence(
            problem_types, math_structures, text
        )
        
        return MathematicalPattern(
            problem_types=problem_types or [ProblemType.UNKNOWN],
            mathematical_structures=math_structures,
            objective_function=objective_function,
            constraints=constraints,
            variables=variables,
            uncertainty_sources=uncertainty_sources,
            iterative_process=iterative_process,
            domain_agnostic_terms=domain_agnostic_terms,
            methodological_keywords=methodological_keywords,
            confidence_score=confidence
        )
    
    @weave.op()
    async def _extract_domain_context(self, text: str) -> DomainContext:
        """Extract domain-specific context from text"""
        
        # Detect primary domain
        primary_domain = "unknown"
        domain_scores = {}
        
        for domain, keywords in self.domain_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            primary_domain = max(domain_scores, key=domain_scores.get)
        
        # Extract domain-specific terms
        domain_specific_terms = []
        if primary_domain in self.domain_patterns:
            domain_specific_terms = [
                term for term in self.domain_patterns[primary_domain] 
                if term in text
            ]
        
        # Extract entities and processes
        entities = self._extract_entities(text)
        processes = self._extract_processes(text)
        
        return DomainContext(
            primary_domain=primary_domain,
            subdomain=None,  # Could be enhanced with more sophisticated detection
            domain_specific_terms=domain_specific_terms,
            entities=entities,
            processes=processes
        )
    
    def _extract_objective_function(self, text: str) -> Optional[str]:
        """Extract objective function description"""
        objective_patterns = [
            r"minimize\s+([^.]+)",
            r"maximize\s+([^.]+)",
            r"optimize\s+([^.]+)",
            r"objective\s+(?:is\s+)?(?:to\s+)?([^.]+)",
            r"goal\s+(?:is\s+)?(?:to\s+)?([^.]+)"
        ]
        
        for pattern in objective_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_constraints(self, text: str) -> List[str]:
        """Extract constraint descriptions"""
        constraint_patterns = [
            r"subject\s+to\s+([^.]+)",
            r"constraint(?:s)?\s*:?\s*([^.]+)",
            r"limited\s+by\s+([^.]+)",
            r"bounded\s+by\s+([^.]+)"
        ]
        
        constraints = []
        for pattern in constraint_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            constraints.extend([match.strip() for match in matches])
        
        return constraints
    
    def _extract_variables(self, text: str) -> List[str]:
        """Extract variable mentions"""
        # Simple heuristic: look for mathematical variable patterns
        variable_patterns = [
            r"\b[xyz]\b",
            r"\b[a-z]\s*\([^)]+\)",
            r"\b(?:parameter|variable)s?\s*([a-zA-Z_]\w*)",
        ]
        
        variables = []
        for pattern in variable_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            variables.extend(matches)
        
        return list(set(variables))
    
    def _extract_uncertainty_sources(self, text: str) -> List[str]:
        """Extract sources of uncertainty"""
        uncertainty_terms = [
            "noise", "uncertainty", "variance", "error", "randomness",
            "stochastic", "probabilistic", "unknown", "ambiguous"
        ]
        
        sources = [term for term in uncertainty_terms if term in text]
        return sources
    
    def _detect_iterative_process(self, text: str) -> bool:
        """Detect if the problem involves iterative processes"""
        iterative_keywords = [
            "iterative", "iterate", "loop", "repeat", "cycle", "update",
            "refine", "improve", "adaptive", "sequential", "stepwise"
        ]
        
        return any(keyword in text for keyword in iterative_keywords)
    
    def _generate_domain_agnostic_terms(self, problem_types: List[ProblemType], 
                                      math_structures: List[MathematicalStructure],
                                      text: str) -> List[str]:
        """Generate search terms that are domain-agnostic"""
        terms = []
        
        # Add problem type terms
        for prob_type in problem_types:
            if prob_type != ProblemType.UNKNOWN:
                try:
                    terms.append(prob_type.value.replace('_', ' '))
                except AttributeError:
                    terms.append(str(prob_type).replace('_', ' '))
        
        # Add mathematical structure terms
        for structure in math_structures:
            try:
                terms.append(structure.value.replace('_', ' '))
            except AttributeError:
                terms.append(str(structure).replace('_', ' '))
        
        # Add common methodological terms
        method_terms = [
            "uncertainty quantification", "adaptive sampling", "iterative refinement",
            "confidence estimation", "robustness analysis", "sensitivity analysis",
            "model validation", "parameter estimation", "feature selection"
        ]
        
        terms.extend([term for term in method_terms if any(word in text for word in term.split())])
        
        return list(set(terms))
    
    def _extract_methodological_keywords(self, text: str) -> List[str]:
        """Extract methodological approach keywords"""
        method_keywords = [
            "machine learning", "deep learning", "neural network", "regression",
            "classification", "clustering", "dimensionality reduction",
            "cross validation", "feature engineering", "ensemble methods",
            "bayesian", "frequentist", "monte carlo", "bootstrap",
            "gradient descent", "genetic algorithm", "particle swarm"
        ]
        
        found_keywords = []
        for keyword in method_keywords:
            if keyword in text:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract domain entities"""
        # Simple noun extraction heuristic
        entity_patterns = [
            r"\b(?:the\s+)?([a-z]+(?:\s+[a-z]+)*)\s+(?:is|are|was|were)",
            r"\b([a-z]+(?:\s+[a-z]+)*)\s+(?:function|model|system|process|method)"
        ]
        
        entities = []
        for pattern in entity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend([match.strip() for match in matches if isinstance(match, str)])
        
        return list(set(entities))
    
    def _extract_processes(self, text: str) -> List[str]:
        """Extract process descriptions"""
        process_patterns = [
            r"(?:process|procedure|method|algorithm)\s+(?:of\s+)?([^.]+)",
            r"([a-z]+ing)\s+(?:process|procedure)",
        ]
        
        processes = []
        for pattern in process_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            processes.extend([match.strip() for match in matches])
        
        return list(set(processes))
    
    def _calculate_pattern_confidence(self, problem_types: List[ProblemType],
                                    math_structures: List[MathematicalStructure],
                                    text: str) -> float:
        """Calculate confidence in the extracted mathematical patterns"""
        confidence = 0.0
        
        # Base confidence from detected patterns
        if problem_types and ProblemType.UNKNOWN not in problem_types:
            confidence += 0.3
        
        if math_structures:
            confidence += 0.3
        
        # Bonus for specific mathematical terms
        math_terms = ["function", "variable", "parameter", "equation", "model"]
        found_terms = sum(1 for term in math_terms if term in text)
        confidence += min(0.2, found_terms * 0.05)
        
        # Bonus for methodological clarity
        method_clarity = len([word for word in text.split() if len(word) > 6]) / len(text.split())
        confidence += min(0.2, method_clarity)
        
        return min(1.0, confidence)
    
    def _calculate_abstraction_confidence(self, mathematical_pattern: MathematicalPattern,
                                        domain_context: DomainContext) -> float:
        """Calculate overall confidence in the problem abstraction"""
        pattern_confidence = mathematical_pattern.confidence_score
        
        # Domain detection confidence
        domain_confidence = 0.5 if domain_context.primary_domain != "unknown" else 0.1
        
        # Term extraction confidence
        term_confidence = min(1.0, len(mathematical_pattern.domain_agnostic_terms) * 0.1)
        
        # Weighted average
        overall_confidence = (
            0.5 * pattern_confidence +
            0.3 * domain_confidence +
            0.2 * term_confidence
        )
        
        return min(1.0, overall_confidence)