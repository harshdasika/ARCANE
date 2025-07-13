"""
Solution Translator - Adapts cross-domain solutions to the target domain
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
import weave

from .problem_abstractor import AbstractedProblem, MathematicalPattern, ProblemType, DomainContext
from .cross_domain_searcher import CrossDomainMatch, SearchResult
from ..config.config_manager import config_manager

logger = logging.getLogger(__name__)

@dataclass
class ConceptMapping:
    """Maps concepts between source and target domains"""
    source_concept: str
    target_concept: str
    confidence: float
    context: str

@dataclass
class MethodologyTranslation:
    """Represents a translated methodology"""
    original_method: str
    translated_method: str
    adaptation_steps: List[str]
    implementation_notes: List[str]
    potential_challenges: List[str]
    confidence: float

@dataclass
class TranslatedSolution:
    """Complete solution translation from source to target domain"""
    source_paper: CrossDomainMatch
    target_domain: str
    concept_mappings: List[ConceptMapping]
    methodology_translation: MethodologyTranslation
    mathematical_adaptations: List[str]
    implementation_roadmap: List[str]
    expected_benefits: List[str]
    potential_limitations: List[str]
    feasibility_score: float
    translation_confidence: float

@dataclass
class TranslationResult:
    """Complete translation results for all cross-domain matches"""
    original_problem: AbstractedProblem
    translated_solutions: List[TranslatedSolution]
    domain_vocabulary: Dict[str, List[str]]
    translation_strategy: Dict[str, Any]
    overall_confidence: float

class SolutionTranslator:
    """Translates cross-domain solutions to the target domain"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_domain_vocabularies()
        self._initialize_translation_rules()
    
    def _initialize_domain_vocabularies(self):
        """Initialize domain-specific vocabularies for translation"""
        
        self.domain_vocabularies = {
            "robotics": {
                "entities": ["robot", "manipulator", "gripper", "sensor", "actuator", "joint", "link"],
                "processes": ["planning", "control", "navigation", "manipulation", "grasping", "trajectory"],
                "measurements": ["position", "velocity", "acceleration", "force", "torque", "orientation"],
                "objectives": ["reach", "grasp", "avoid", "track", "stabilize", "optimize path"],
                "constraints": ["collision", "joint limits", "workspace", "dynamics", "kinematic"],
                "uncertainty": ["sensor noise", "model error", "disturbance", "calibration error"]
            },
            "biomolecular": {
                "entities": ["molecule", "protein", "drug", "compound", "receptor", "binding site", "sequence"],
                "processes": ["design", "synthesis", "binding", "folding", "interaction", "docking"],
                "measurements": ["affinity", "selectivity", "stability", "activity", "concentration", "structure"],
                "objectives": ["optimize binding", "improve selectivity", "enhance stability", "reduce toxicity"],
                "constraints": ["chemical feasibility", "synthesis difficulty", "bioavailability", "side effects"],
                "uncertainty": ["experimental error", "structural flexibility", "binding variability", "assay noise"]
            },
            "computer_vision": {
                "entities": ["image", "pixel", "feature", "object", "region", "detector", "classifier"],
                "processes": ["detection", "recognition", "segmentation", "tracking", "classification", "learning"],
                "measurements": ["accuracy", "precision", "recall", "IoU", "confidence", "loss"],
                "objectives": ["detect objects", "classify images", "segment regions", "track motion"],
                "constraints": ["computational cost", "real-time", "memory", "accuracy requirements"],
                "uncertainty": ["lighting variation", "occlusion", "noise", "viewpoint changes"]
            },
            "machine_learning": {
                "entities": ["model", "network", "layer", "neuron", "parameter", "feature", "dataset"],
                "processes": ["training", "learning", "inference", "optimization", "regularization", "validation"],
                "measurements": ["accuracy", "loss", "gradient", "weight", "activation", "performance"],
                "objectives": ["minimize error", "maximize accuracy", "improve generalization", "reduce overfitting"],
                "constraints": ["computational budget", "data availability", "model complexity", "interpretability"],
                "uncertainty": ["data noise", "model uncertainty", "epistemic", "aleatoric", "generalization error"]
            },
            "physics": {
                "entities": ["particle", "field", "system", "state", "observable", "operator", "potential"],
                "processes": ["evolution", "interaction", "measurement", "transition", "decay", "scattering"],
                "measurements": ["energy", "momentum", "probability", "amplitude", "cross-section", "lifetime"],
                "objectives": ["minimize energy", "conserve quantity", "achieve equilibrium", "optimize coupling"],
                "constraints": ["conservation laws", "symmetry", "boundary conditions", "physical limits"],
                "uncertainty": ["quantum uncertainty", "measurement error", "statistical fluctuation", "systematic error"]
            },
            "materials": {
                "entities": ["material", "crystal", "alloy", "structure", "grain", "defect", "interface"],
                "processes": ["synthesis", "characterization", "processing", "annealing", "deformation", "growth"],
                "measurements": ["strength", "conductivity", "stiffness", "density", "composition", "microstructure"],
                "objectives": ["optimize properties", "improve performance", "reduce cost", "enhance durability"],
                "constraints": ["processing limits", "thermodynamic stability", "cost", "environmental impact"],
                "uncertainty": ["processing variation", "measurement error", "structural disorder", "composition variation"]
            },
            "finance": {
                "entities": ["asset", "portfolio", "security", "derivative", "market", "trader", "position"],
                "processes": ["trading", "optimization", "hedging", "pricing", "risk management", "allocation"],
                "measurements": ["return", "volatility", "risk", "correlation", "value", "exposure"],
                "objectives": ["maximize return", "minimize risk", "optimize portfolio", "hedge exposure"],
                "constraints": ["capital limits", "regulatory", "liquidity", "transaction costs", "market impact"],
                "uncertainty": ["market volatility", "model risk", "liquidity risk", "credit risk", "operational risk"]
            }
        }
        
        # Common mathematical concepts across domains
        self.mathematical_concepts = {
            "optimization": ["minimize", "maximize", "optimal", "objective", "constraint"],
            "uncertainty": ["probability", "confidence", "variance", "risk", "error"],
            "learning": ["adapt", "update", "train", "learn", "improve"],
            "prediction": ["forecast", "estimate", "predict", "model", "infer"],
            "control": ["regulate", "stabilize", "feedback", "control", "adjust"]
        }
    
    def _initialize_translation_rules(self):
        """Initialize rules for translating concepts between domains"""
        
        # Common concept mappings between domains
        self.concept_mappings = {
            ("biomolecular", "robotics"): {
                "molecular design": "motion planning",
                "protein folding": "robot configuration",
                "binding affinity": "grasp quality",
                "drug discovery": "optimal control",
                "molecular dynamics": "robot dynamics",
                "active site": "end effector",
                "conformational search": "path planning",
                "chemical space": "configuration space"
            },
            ("robotics", "biomolecular"): {
                "motion planning": "molecular design",
                "robot configuration": "protein conformation",
                "grasp quality": "binding affinity",
                "optimal control": "drug optimization",
                "robot dynamics": "molecular dynamics",
                "end effector": "active site",
                "path planning": "conformational search",
                "configuration space": "chemical space"
            },
            ("machine_learning", "robotics"): {
                "neural network": "control system",
                "training data": "demonstration trajectories",
                "model accuracy": "control performance",
                "overfitting": "overtuning",
                "feature extraction": "sensor processing",
                "classification": "state estimation",
                "reinforcement learning": "adaptive control"
            },
            ("finance", "robotics"): {
                "portfolio optimization": "trajectory optimization",
                "risk management": "uncertainty handling",
                "asset allocation": "resource allocation",
                "market volatility": "environmental uncertainty",
                "trading strategy": "control policy",
                "return maximization": "performance optimization"
            }
        }
        
        # Process translation patterns
        self.process_translations = {
            "iterative_optimization": {
                "robotics": "iterative motion refinement",
                "biomolecular": "iterative molecular optimization",
                "finance": "adaptive portfolio rebalancing",
                "materials": "iterative property optimization"
            },
            "uncertainty_quantification": {
                "robotics": "motion uncertainty analysis",
                "biomolecular": "binding uncertainty estimation",
                "finance": "risk assessment",
                "materials": "property uncertainty characterization"
            },
            "active_learning": {
                "robotics": "adaptive trajectory learning",
                "biomolecular": "active molecular design",
                "finance": "adaptive market learning",
                "materials": "guided materials discovery"
            }
        }
    
    @weave.op()
    async def translate_solutions(self, search_result: SearchResult) -> TranslationResult:
        """
        Main method to translate cross-domain solutions to the target domain
        
        Args:
            search_result: Results from cross-domain search
            
        Returns:
            TranslationResult containing all translated solutions
        """
        try:
            self.logger.info("Starting solution translation")
            
            target_domain = search_result.original_problem.domain_context.primary_domain
            
            # Translate each cross-domain match
            translated_solutions = []
            for match in search_result.matches:
                if match.domain != target_domain:  # Only translate cross-domain solutions
                    translation = await self._translate_single_solution(
                        search_result.original_problem, match, target_domain
                    )
                    if translation:
                        translated_solutions.append(translation)
            
            # Build domain vocabulary for reference
            domain_vocabulary = self._build_domain_vocabulary(target_domain, translated_solutions)
            
            # Create translation strategy summary
            translation_strategy = self._create_translation_strategy(
                search_result.original_problem, translated_solutions
            )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(translated_solutions)
            
            translation_result = TranslationResult(
                original_problem=search_result.original_problem,
                translated_solutions=translated_solutions,
                domain_vocabulary=domain_vocabulary,
                translation_strategy=translation_strategy,
                overall_confidence=overall_confidence
            )
            
            self.logger.info(f"Translation completed: {len(translated_solutions)} solutions translated")
            return translation_result
            
        except Exception as e:
            self.logger.error(f"Error in solution translation: {str(e)}")
            raise
    
    @weave.op()
    async def _translate_single_solution(self, original_problem: AbstractedProblem,
                                       match: CrossDomainMatch, 
                                       target_domain: str) -> Optional[TranslatedSolution]:
        """Translate a single cross-domain solution to the target domain"""
        
        source_domain = match.domain
        
        # Generate concept mappings
        concept_mappings = await self._generate_concept_mappings(
            source_domain, target_domain, match, original_problem
        )
        
        # Translate methodology
        methodology_translation = await self._translate_methodology(
            match, target_domain, original_problem.mathematical_pattern
        )
        
        # Generate mathematical adaptations
        mathematical_adaptations = await self._generate_mathematical_adaptations(
            match, original_problem.mathematical_pattern, target_domain
        )
        
        # Create implementation roadmap
        implementation_roadmap = await self._create_implementation_roadmap(
            match, target_domain, methodology_translation, concept_mappings
        )
        
        # Identify expected benefits and limitations
        expected_benefits = self._identify_expected_benefits(match, target_domain)
        potential_limitations = self._identify_potential_limitations(match, target_domain)
        
        # Calculate feasibility and confidence scores
        feasibility_score = self._calculate_feasibility_score(
            match, target_domain, concept_mappings, methodology_translation
        )
        
        translation_confidence = self._calculate_translation_confidence(
            concept_mappings, methodology_translation, mathematical_adaptations
        )
        
        return TranslatedSolution(
            source_paper=match,
            target_domain=target_domain,
            concept_mappings=concept_mappings,
            methodology_translation=methodology_translation,
            mathematical_adaptations=mathematical_adaptations,
            implementation_roadmap=implementation_roadmap,
            expected_benefits=expected_benefits,
            potential_limitations=potential_limitations,
            feasibility_score=feasibility_score,
            translation_confidence=translation_confidence
        )
    
    @weave.op()
    async def _generate_concept_mappings(self, source_domain: str, target_domain: str,
                                       match: CrossDomainMatch, 
                                       original_problem: AbstractedProblem) -> List[ConceptMapping]:
        """Generate mappings between source and target domain concepts"""
        
        mappings = []
        
        # Use predefined concept mappings if available
        mapping_key = (source_domain, target_domain)
        if mapping_key in self.concept_mappings:
            predefined_mappings = self.concept_mappings[mapping_key]
            
            # Find relevant mappings based on paper content
            paper_text = ((match.title or "") + " " + (match.abstract or "")).lower()
            
            for source_concept, target_concept in predefined_mappings.items():
                if any(word in paper_text for word in source_concept.split()):
                    mappings.append(ConceptMapping(
                        source_concept=source_concept,
                        target_concept=target_concept,
                        confidence=0.8,
                        context=f"Standard mapping from {source_domain} to {target_domain}"
                    ))
        
        # Generate domain-specific vocabulary mappings
        if source_domain in self.domain_vocabularies and target_domain in self.domain_vocabularies:
            source_vocab = self.domain_vocabularies[source_domain]
            target_vocab = self.domain_vocabularies[target_domain]
            
            # Map entities
            mappings.extend(self._map_vocabulary_categories(
                source_vocab["entities"], target_vocab["entities"], 
                "entity", match, 0.6
            ))
            
            # Map processes
            mappings.extend(self._map_vocabulary_categories(
                source_vocab["processes"], target_vocab["processes"],
                "process", match, 0.7
            ))
            
            # Map objectives
            mappings.extend(self._map_vocabulary_categories(
                source_vocab["objectives"], target_vocab["objectives"],
                "objective", match, 0.8
            ))
        
        # Generate mathematical concept mappings
        math_mappings = self._generate_mathematical_mappings(
            original_problem.mathematical_pattern, match, target_domain
        )
        mappings.extend(math_mappings)
        
        return mappings[:10]  # Limit to most relevant mappings
    
    def _map_vocabulary_categories(self, source_terms: List[str], target_terms: List[str],
                                 category: str, match: CrossDomainMatch, 
                                 base_confidence: float) -> List[ConceptMapping]:
        """Map vocabulary terms between categories"""
        
        mappings = []
        paper_text = ((match.title or "") + " " + (match.abstract or "")).lower()
        
        for source_term in source_terms:
            if source_term in paper_text:
                # Find best target mapping (simplified heuristic)
                best_target = self._find_best_vocabulary_match(source_term, target_terms)
                if best_target:
                    mappings.append(ConceptMapping(
                        source_concept=source_term,
                        target_concept=best_target,
                        confidence=base_confidence,
                        context=f"{category.capitalize()} mapping"
                    ))
        
        return mappings
    
    def _find_best_vocabulary_match(self, source_term: str, target_terms: List[str]) -> Optional[str]:
        """Find the best matching target term for a source term"""
        
        # Simple heuristic: return first available target term
        # In a more sophisticated implementation, this could use semantic similarity
        if target_terms:
            return target_terms[0]
        return None
    
    def _generate_mathematical_mappings(self, pattern: MathematicalPattern,
                                      match: CrossDomainMatch, 
                                      target_domain: str) -> List[ConceptMapping]:
        """Generate mappings for mathematical concepts"""
        
        mappings = []
        paper_text = ((match.title or "") + " " + (match.abstract or "")).lower()
        
        # Map mathematical structures to domain-specific implementations
        for structure in pattern.mathematical_structures:
            struct_name = structure.value.replace('_', ' ')
            
            if any(word in paper_text for word in struct_name.split()):
                domain_implementation = self._get_domain_math_implementation(structure, target_domain)
                if domain_implementation:
                    mappings.append(ConceptMapping(
                        source_concept=struct_name,
                        target_concept=domain_implementation,
                        confidence=0.7,
                        context="Mathematical structure mapping"
                    ))
        
        return mappings
    
    def _get_domain_math_implementation(self, structure, target_domain: str) -> Optional[str]:
        """Get domain-specific implementation of mathematical structure"""
        
        implementations = {
            "robotics": {
                "optimization_theory": "trajectory optimization",
                "probability_theory": "uncertainty propagation",
                "linear_algebra": "kinematic transformations",
                "differential_equations": "robot dynamics modeling"
            },
            "biomolecular": {
                "optimization_theory": "molecular optimization",
                "probability_theory": "binding probability estimation",
                "graph_theory": "molecular network analysis",
                "information_theory": "sequence information content"
            }
        }
        
        domain_impls = implementations.get(target_domain, {})
        return domain_impls.get(structure.value)
    
    @weave.op()
    async def _translate_methodology(self, match: CrossDomainMatch, target_domain: str,
                                   pattern: MathematicalPattern) -> MethodologyTranslation:
        """Translate the core methodology to the target domain"""
        
        # Extract core methodology from paper
        original_method = self._extract_core_methodology(match)
        
        # Translate to target domain
        translated_method = self._adapt_methodology_to_domain(original_method, target_domain)
        
        # Generate adaptation steps
        adaptation_steps = self._generate_adaptation_steps(
            original_method, translated_method, target_domain, pattern
        )
        
        # Create implementation notes
        implementation_notes = self._generate_implementation_notes(
            translated_method, target_domain, pattern
        )
        
        # Identify potential challenges
        potential_challenges = self._identify_methodology_challenges(
            original_method, target_domain, pattern
        )
        
        # Calculate confidence
        confidence = self._calculate_methodology_confidence(
            original_method, translated_method, target_domain
        )
        
        return MethodologyTranslation(
            original_method=original_method,
            translated_method=translated_method,
            adaptation_steps=adaptation_steps,
            implementation_notes=implementation_notes,
            potential_challenges=potential_challenges,
            confidence=confidence
        )
    
    def _extract_core_methodology(self, match: CrossDomainMatch) -> str:
        """Extract the core methodological approach from the paper"""
        
        # Simple extraction based on matching patterns and abstract
        methodology_indicators = [
            "method", "approach", "algorithm", "technique", "framework",
            "procedure", "strategy", "scheme", "process"
        ]
        
        abstract = (match.abstract or "").lower()
        title = (match.title or "").lower()
        
        # Look for methodology descriptions in abstract
        sentences = abstract.split('.')
        method_sentences = []
        
        for sentence in sentences:
            if any(indicator in sentence for indicator in methodology_indicators):
                method_sentences.append(sentence.strip())
        
        if method_sentences:
            return '. '.join(method_sentences[:2])  # First 2 relevant sentences
        
        # Fallback to title + matching patterns
        patterns_text = ', '.join(match.matching_patterns)
        return f"Method based on {title} using {patterns_text}"
    
    def _adapt_methodology_to_domain(self, original_method: str, target_domain: str) -> str:
        """Adapt the methodology description to the target domain"""
        
        if target_domain not in self.domain_vocabularies:
            return original_method
        
        adapted_method = original_method
        target_vocab = self.domain_vocabularies[target_domain]
        
        # Replace generic terms with domain-specific ones
        entities = target_vocab.get("entities", []) or []
        generic_replacements = {
            "system": entities[0] if entities else "system",
            "optimization": f"{target_domain} optimization",
            "performance": f"{target_domain} performance",
            "model": f"{target_domain} model"
        }
        
        for generic, specific in generic_replacements.items():
            adapted_method = adapted_method.replace(generic, specific)
        
        return adapted_method
    
    def _generate_adaptation_steps(self, original_method: str, translated_method: str,
                                 target_domain: str, pattern: MathematicalPattern) -> List[str]:
        """Generate concrete steps for adapting the methodology"""
        
        steps = []
        
        # Step 1: Domain contextualization
        steps.append(f"Contextualize the approach within {target_domain} framework")
        
        # Step 2: Data/input adaptation
        if target_domain in self.domain_vocabularies:
            entities = self.domain_vocabularies[target_domain].get("entities", []) or []
            if entities:
                steps.append(f"Adapt input data to work with {entities[0]} characteristics")
        
        # Step 3: Mathematical adaptation
        math_structures = pattern.mathematical_structures or []
        if math_structures:
            structures = []
            for s in math_structures[:2]:
                try:
                    structures.append(s.value.replace('_', ' '))
                except AttributeError:
                    structures.append(str(s).replace('_', ' '))
            steps.append(f"Implement {', '.join(structures)} for {target_domain} domain")
        
        # Step 4: Objective function adaptation
        if pattern.objective_function:
            steps.append(f"Reformulate objective function for {target_domain} goals")
        
        # Step 5: Validation adaptation
        steps.append(f"Design {target_domain}-specific validation and evaluation metrics")
        
        return steps
    
    def _generate_implementation_notes(self, translated_method: str, target_domain: str,
                                     pattern: MathematicalPattern) -> List[str]:
        """Generate implementation-specific notes"""
        
        notes = []
        
        # Domain-specific implementation considerations
        if target_domain in self.domain_vocabularies:
            constraints = self.domain_vocabularies[target_domain].get("constraints", []) or []
            if constraints:
                notes.append(f"Consider {target_domain} constraints: {', '.join(constraints[:3])}")
        
        # Mathematical implementation notes
        problem_types = pattern.problem_types or []
        if ProblemType.UNCERTAINTY_QUANTIFICATION in problem_types:
            notes.append("Implement uncertainty quantification appropriate for the domain")
        
        if pattern.iterative_process:
            notes.append("Design iterative refinement loop with domain-appropriate stopping criteria")
        
        # Computational considerations
        notes.append("Consider computational requirements and real-time constraints if applicable")
        
        # Validation considerations
        notes.append(f"Plan validation using {target_domain}-standard evaluation protocols")
        
        return notes
    
    def _identify_methodology_challenges(self, original_method: str, target_domain: str,
                                       pattern: MathematicalPattern) -> List[str]:
        """Identify potential challenges in methodology translation"""
        
        challenges = []
        
        # Domain-specific challenges
        if target_domain in self.domain_vocabularies:
            uncertainties = self.domain_vocabularies[target_domain].get("uncertainty", [])
            if uncertainties:
                challenges.append(f"Handle {target_domain}-specific uncertainties: {uncertainties[0]}")
        
        # Mathematical challenges
        if len(pattern.mathematical_structures) > 2:
            challenges.append("Complex mathematical requirements may need simplification")
        
        if pattern.uncertainty_sources:
            challenges.append("Uncertainty modeling may require domain-specific adaptations")
        
        # Data challenges
        challenges.append(f"Availability and quality of {target_domain} data for training/validation")
        
        # Validation challenges
        challenges.append(f"Establishing appropriate success metrics for {target_domain}")
        
        return challenges
    
    def _calculate_methodology_confidence(self, original_method: str, 
                                        translated_method: str, target_domain: str) -> float:
        """Calculate confidence in methodology translation"""
        
        confidence = 0.5  # Base confidence
        
        # Boost for well-defined original method
        if len(original_method.split()) > 10:
            confidence += 0.2
        
        # Boost for domain vocabulary availability
        if target_domain in self.domain_vocabularies:
            confidence += 0.2
        
        # Boost for successful translation (non-identical result)
        if translated_method != original_method:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    @weave.op()
    async def _generate_mathematical_adaptations(self, match: CrossDomainMatch,
                                               pattern: MathematicalPattern,
                                               target_domain: str) -> List[str]:
        """Generate mathematical adaptations needed for the target domain"""
        
        adaptations = []
        
        # Adaptation for each mathematical structure
        for structure in pattern.mathematical_structures:
            adaptation = self._get_structure_adaptation(structure, target_domain)
            if adaptation:
                adaptations.append(adaptation)
        
        # Problem-type specific adaptations
        for prob_type in pattern.problem_types:
            if prob_type != ProblemType.UNKNOWN:
                adaptation = self._get_problem_type_adaptation(prob_type, target_domain)
                if adaptation:
                    adaptations.append(adaptation)
        
        # Domain-specific mathematical considerations
        if target_domain in self.domain_vocabularies:
            measurements = self.domain_vocabularies[target_domain].get("measurements", [])
            if measurements:
                adaptations.append(f"Adapt mathematical formulation to work with {measurements[0]} measurements")
        
        return adaptations[:5]  # Limit to 5 key adaptations
    
    def _get_structure_adaptation(self, structure, target_domain: str) -> Optional[str]:
        """Get adaptation for specific mathematical structure"""
        
        structure_adaptations = {
            "robotics": {
                "optimization_theory": "Formulate as trajectory optimization with kinematic/dynamic constraints",
                "probability_theory": "Model uncertainty in robot state estimation and control",
                "linear_algebra": "Utilize robot kinematics and transformation matrices"
            },
            "biomolecular": {
                "optimization_theory": "Formulate molecular design as constrained optimization problem",
                "probability_theory": "Model binding affinity and selectivity uncertainties",
                "graph_theory": "Represent molecular structures and interaction networks"
            }
        }
        
        domain_adaptations = structure_adaptations.get(target_domain, {})
        return domain_adaptations.get(structure.value)
    
    def _get_problem_type_adaptation(self, prob_type: ProblemType, target_domain: str) -> Optional[str]:
        """Get adaptation for specific problem type"""
        
        type_adaptations = {
            "robotics": {
                ProblemType.OPTIMIZATION: "Optimize robot trajectories and control policies",
                ProblemType.UNCERTAINTY_QUANTIFICATION: "Quantify motion and sensing uncertainties",
                ProblemType.ACTIVE_LEARNING: "Learn robot policies through active exploration"
            },
            "biomolecular": {
                ProblemType.OPTIMIZATION: "Optimize molecular properties and binding characteristics",
                ProblemType.UNCERTAINTY_QUANTIFICATION: "Quantify molecular design uncertainties",
                ProblemType.ACTIVE_LEARNING: "Guide molecular experiments through active design"
            }
        }
        
        domain_adaptations = type_adaptations.get(target_domain, {})
        return domain_adaptations.get(prob_type)
    
    @weave.op()
    async def _create_implementation_roadmap(self, match: CrossDomainMatch, target_domain: str,
                                           methodology: MethodologyTranslation,
                                           mappings: List[ConceptMapping]) -> List[str]:
        """Create a step-by-step implementation roadmap"""
        
        roadmap = []
        
        # Phase 1: Foundation
        roadmap.append("Phase 1 - Foundation Setup:")
        roadmap.append(f"  • Study original paper: {match.title}")
        roadmap.append(f"  • Establish {target_domain} development environment")
        roadmap.append("  • Identify required mathematical libraries and tools")
        
        # Phase 2: Adaptation
        roadmap.append("Phase 2 - Methodology Adaptation:")
        roadmap.append(f"  • Implement core methodology: {methodology.original_method[:50]}...")
        roadmap.append(f"  • Adapt for {target_domain} constraints and requirements")
        if mappings:
            key_mapping = mappings[0]
            roadmap.append(f"  • Map key concepts (e.g., {key_mapping.source_concept} → {key_mapping.target_concept})")
        
        # Phase 3: Implementation
        roadmap.append("Phase 3 - Core Implementation:")
        for i, step in enumerate(methodology.adaptation_steps[:3], 1):
            roadmap.append(f"  • Step {i}: {step}")
        
        # Phase 4: Validation
        roadmap.append("Phase 4 - Validation and Testing:")
        roadmap.append(f"  • Design {target_domain}-appropriate test cases")
        roadmap.append("  • Validate against known benchmarks or baselines")
        roadmap.append("  • Compare performance with existing domain methods")
        
        # Phase 5: Optimization
        roadmap.append("Phase 5 - Optimization and Refinement:")
        roadmap.append(f"  • Optimize for {target_domain}-specific performance metrics")
        roadmap.append("  • Address identified challenges and limitations")
        roadmap.append("  • Document methodology and create reproducible implementation")
        
        return roadmap
    
    def _identify_expected_benefits(self, match: CrossDomainMatch, target_domain: str) -> List[str]:
        """Identify expected benefits of applying this solution"""
        
        benefits = []
        
        # Benefits based on citation count (proxy for impact)
        high_impact_threshold = config_manager.get('temporal', 'citation_thresholds.high_impact', 50)
        if match.citation_count > high_impact_threshold:
            benefits.append("Proven methodology with strong research impact")
        
        # Benefits based on similarity score
        if match.similarity_score.overall_similarity > 0.7:
            benefits.append("High methodological similarity suggests strong applicability")
        
        # Domain-specific benefits
        benefits.append(f"Novel approach not commonly used in {target_domain}")
        benefits.append("Potential for significant performance improvements")
        benefits.append("Cross-domain insights may reveal new research directions")
        
        # Mathematical benefits
        if match.similarity_score.mathematical_similarity > 0.6:
            benefits.append("Strong mathematical foundation ensures theoretical soundness")
        
        return benefits
    
    def _identify_potential_limitations(self, match: CrossDomainMatch, target_domain: str) -> List[str]:
        """Identify potential limitations and challenges"""
        
        limitations = []
        
        # Confidence-based limitations
        if match.similarity_score.confidence < 0.6:
            limitations.append("Moderate confidence in similarity assessment")
        
        # Domain gap limitations
        if match.domain != "unknown":
            limitations.append(f"Significant domain gap between {match.domain} and {target_domain}")
        
        # Implementation limitations
        limitations.append("May require significant adaptation and domain expertise")
        limitations.append("Validation may be challenging due to domain differences")
        
        # Data limitations
        limitations.append(f"May require {target_domain}-specific data that differs from original")
        
        # Complexity limitations
        if len(match.matching_patterns) < 3:
            limitations.append("Limited pattern matching may indicate adaptation challenges")
        
        return limitations
    
    def _calculate_feasibility_score(self, match: CrossDomainMatch, target_domain: str,
                                   mappings: List[ConceptMapping],
                                   methodology: MethodologyTranslation) -> float:
        """Calculate feasibility score for implementing this solution"""
        
        score = 0.0
        
        # Base score from similarity
        score += 0.3 * match.similarity_score.overall_similarity
        
        # Boost for strong concept mappings
        if mappings:
            avg_mapping_confidence = sum(m.confidence for m in mappings) / len(mappings)
            score += 0.2 * avg_mapping_confidence
        
        # Boost for methodology confidence
        score += 0.2 * methodology.confidence
        
        # Boost for domain vocabulary availability
        if target_domain in self.domain_vocabularies:
            score += 0.1
        
        # Boost for high-impact source paper
        medium_impact_threshold = config_manager.get('temporal', 'citation_thresholds.medium_impact', 20)
        if match.citation_count > medium_impact_threshold:
            score += 0.1
        
        # Penalty for old papers (may be outdated)
        outdated_year_threshold = config_manager.get('temporal', 'paper_age.outdated_threshold', 2015)
        if match.year and match.year < outdated_year_threshold:
            score -= 0.1
        
        # Boost for recent papers
        if match.year and match.year > 2020:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_translation_confidence(self, mappings: List[ConceptMapping],
                                        methodology: MethodologyTranslation,
                                        adaptations: List[str]) -> float:
        """Calculate overall confidence in the translation"""
        
        confidence = 0.0
        
        # Concept mapping confidence
        if mappings:
            avg_mapping_confidence = sum(m.confidence for m in mappings) / len(mappings)
            confidence += 0.4 * avg_mapping_confidence
        else:
            confidence += 0.1  # Low confidence for no mappings
        
        # Methodology translation confidence
        confidence += 0.4 * methodology.confidence
        
        # Adaptation completeness
        adaptation_completeness = min(1.0, len(adaptations) / 5)  # Expect ~5 adaptations
        confidence += 0.2 * adaptation_completeness
        
        return min(1.0, confidence)
    
    def _build_domain_vocabulary(self, target_domain: str, 
                               solutions: List[TranslatedSolution]) -> Dict[str, List[str]]:
        """Build domain vocabulary from translated solutions"""
        
        vocabulary = {
            "entities": [],
            "processes": [],
            "measurements": [],
            "objectives": [],
            "constraints": [],
            "uncertainty": []
        }
        
        # Collect vocabulary from all solutions
        for solution in solutions:
            for mapping in solution.concept_mappings:
                if mapping.target_concept:
                    vocabulary["entities"].append(mapping.target_concept)
        
        # Remove duplicates and limit
        for key in vocabulary:
            vocabulary[key] = list(dict.fromkeys(vocabulary[key]))[:10]
        
        return vocabulary
    
    def _create_translation_strategy(self, original_problem: AbstractedProblem,
                                   solutions: List[TranslatedSolution]) -> Dict[str, Any]:
        """Create a summary of the translation strategy used"""
        
        strategy = {
            "target_domain": original_problem.domain_context.primary_domain,
            "solutions_translated": len(solutions),
            "source_domains": list(set(s.source_paper.domain for s in solutions)),
            "translation_confidence": sum(s.translation_confidence for s in solutions) / len(solutions) if solutions else 0.0,
            "average_feasibility": sum(s.feasibility_score for s in solutions) / len(solutions) if solutions else 0.0
        }
        
        # Collect key challenges and benefits
        all_challenges = []
        all_benefits = []
        
        for solution in solutions:
            all_challenges.extend(solution.potential_limitations)
            all_benefits.extend(solution.expected_benefits)
        
        strategy["key_challenges"] = list(dict.fromkeys(all_challenges))[:5]
        strategy["success_factors"] = list(dict.fromkeys(all_benefits))[:5]
        
        return strategy
    
    def _calculate_overall_confidence(self, solutions: List[TranslatedSolution]) -> float:
        """Calculate overall confidence in all translations"""
        
        if not solutions:
            return 0.1
        
        # Average of individual translation confidences
        avg_confidence = sum(s.translation_confidence for s in solutions) / len(solutions)
        
        # Boost for multiple high-quality solutions
        high_quality_count = sum(1 for s in solutions if s.feasibility_score > 0.6)
        diversity_bonus = min(0.2, high_quality_count * 0.05)
        
        return min(1.0, avg_confidence + diversity_bonus)