"""
Reasoning Tracer - Tracks and traces the entire cross-domain discovery process using W&B Weave
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import weave

from .problem_abstractor import AbstractedProblem, MathematicalPattern
from .cross_domain_searcher import SearchResult, CrossDomainMatch
from .solution_translator import TranslationResult, TranslatedSolution
from ..config.config_manager import config_manager

logger = logging.getLogger(__name__)

@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning process"""
    step_id: str
    step_name: str
    description: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    confidence: float
    execution_time: float
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class DecisionPoint:
    """Represents a decision made during the process"""
    decision_id: str
    description: str
    options_considered: List[str]
    chosen_option: str
    reasoning: str
    confidence: float
    impact_on_process: str

@dataclass
class SimilarityJustification:
    """Detailed justification for similarity scores"""
    paper_id: str
    paper_title: str
    overall_similarity: float
    component_scores: Dict[str, float]
    matching_evidence: List[str]
    reasoning_chain: List[str]

@dataclass
class ProcessTrace:
    """Complete trace of the cross-domain discovery process"""
    session_id: str
    original_problem_description: str
    start_time: str
    end_time: Optional[str]
    reasoning_steps: List[ReasoningStep]
    decision_points: List[DecisionPoint]
    similarity_justifications: List[SimilarityJustification]
    key_insights: List[str]
    methodology_transfers: List[Dict[str, Any]]
    success_metrics: Dict[str, float]
    total_execution_time: float

class ReasoningTracer:
    """Traces and documents the entire reasoning process for transparency"""
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize reasoning tracer
        
        Args:
            session_id: Unique identifier for this reasoning session
        """
        self.logger = logging.getLogger(__name__)
        self.session_id = session_id or f"session_{int(time.time())}"
        self.start_time = datetime.now().isoformat()
        
        # Initialize trace components
        self.reasoning_steps: List[ReasoningStep] = []
        self.decision_points: List[DecisionPoint] = []
        self.similarity_justifications: List[SimilarityJustification] = []
        self.key_insights: List[str] = []
        self.methodology_transfers: List[Dict[str, Any]] = []
        
        # Process state
        self.current_step = 0
        self.original_problem_description = ""
        
        self.logger.info(f"Reasoning tracer initialized for session: {self.session_id}")
    
    @weave.op()
    def start_session(self, problem_description: str) -> str:
        """
        Start a new reasoning session
        
        Args:
            problem_description: Original research problem description
            
        Returns:
            Session ID for tracking
        """
        self.original_problem_description = problem_description
        
        # Note: Weave tracing is handled automatically via @weave.op() decorators
        # Session data is captured in the op traces
        
        self.logger.info(f"Started reasoning session: {self.session_id}")
        return self.session_id
    
    @weave.op()
    def trace_abstraction_step(self, problem_description: str, 
                              abstracted_problem: AbstractedProblem,
                              execution_time: float) -> None:
        """
        Trace the problem abstraction step
        
        Args:
            problem_description: Original problem description
            abstracted_problem: Result of abstraction
            execution_time: Time taken for abstraction
        """
        step_id = f"step_{self.current_step:03d}_abstraction"
        self.current_step += 1
        
        # Create reasoning step
        reasoning_step = ReasoningStep(
            step_id=step_id,
            step_name="Problem Abstraction",
            description="Extract mathematical patterns and domain context from research problem",
            inputs={"problem_description": problem_description},
            outputs={
                "mathematical_patterns": [p.value for p in abstracted_problem.mathematical_pattern.problem_types],
                "mathematical_structures": [s.value for s in abstracted_problem.mathematical_pattern.mathematical_structures],
                "domain": abstracted_problem.domain_context.primary_domain,
                "confidence": abstracted_problem.abstraction_confidence,
                "domain_agnostic_terms": abstracted_problem.mathematical_pattern.domain_agnostic_terms
            },
            confidence=abstracted_problem.abstraction_confidence,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            metadata={
                "iterative_process": abstracted_problem.mathematical_pattern.iterative_process,
                "uncertainty_sources": abstracted_problem.mathematical_pattern.uncertainty_sources,
                "methodological_keywords": abstracted_problem.mathematical_pattern.methodological_keywords
            }
        )
        
        self.reasoning_steps.append(reasoning_step)
        
        # Weave tracing captures this data automatically via @weave.op() decorator
        
        # Record key insights from abstraction
        insights = self._extract_abstraction_insights(abstracted_problem)
        self.key_insights.extend(insights)
        
        self.logger.info(f"Traced abstraction step: {step_id}")
    
    @weave.op()
    def trace_search_step(self, abstracted_problem: AbstractedProblem,
                         search_result: SearchResult,
                         execution_time: float) -> None:
        """
        Trace the cross-domain search step
        
        Args:
            abstracted_problem: Abstracted problem used for search
            search_result: Results from cross-domain search
            execution_time: Time taken for search
        """
        step_id = f"step_{self.current_step:03d}_search"
        self.current_step += 1
        
        # Get top similarity scores safely
        top_similarity_scores = []
        if search_result.matches:
            top_matches = search_result.matches[:5]
            top_similarity_scores = [m.similarity_score.overall_similarity for m in top_matches]
        
        # Create reasoning step
        reasoning_step = ReasoningStep(
            step_id=step_id,
            step_name="Cross-Domain Search",
            description="Search for structurally similar solutions across scientific domains",
            inputs={
                "search_terms": abstracted_problem.mathematical_pattern.domain_agnostic_terms,
                "target_domains": search_result.domains_searched,
                "mathematical_patterns": [p.value for p in abstracted_problem.mathematical_pattern.problem_types]
            },
            outputs={
                "papers_found": len(search_result.matches),
                "domains_covered": search_result.domains_searched,
                "papers_analyzed": search_result.total_papers_analyzed,
                "search_confidence": search_result.search_confidence,
                "top_similarity_scores": top_similarity_scores
            },
            confidence=search_result.search_confidence,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            metadata={
                "search_strategy": search_result.search_strategy,
                "unique_domains": len(set(m.domain for m in search_result.matches))
            }
        )
        
        self.reasoning_steps.append(reasoning_step)
        
        # Weave tracing captures this data automatically via @weave.op() decorator
        
        # Trace similarity justifications for top matches
        if search_result.matches:
            top_matches = search_result.matches[:5]
            self._trace_similarity_justifications(top_matches)
        
        # Record search insights
        insights = self._extract_search_insights(search_result)
        self.key_insights.extend(insights)
        
        self.logger.info(f"Traced search step: {step_id}")
    
    @weave.op()
    def trace_translation_step(self, search_result: SearchResult,
                              translation_result: TranslationResult,
                              execution_time: float) -> None:
        """
        Trace the solution translation step
        
        Args:
            search_result: Search results used for translation
            translation_result: Translation results
            execution_time: Time taken for translation
        """
        step_id = f"step_{self.current_step:03d}_translation"
        self.current_step += 1
        
        # Create reasoning step
        reasoning_step = ReasoningStep(
            step_id=step_id,
            step_name="Solution Translation",
            description="Translate cross-domain solutions to target domain",
            inputs={
                "source_papers": len(search_result.matches),
                "target_domain": translation_result.original_problem.domain_context.primary_domain,
                "source_domains": list(set(m.domain for m in search_result.matches))
            },
            outputs={
                "translated_solutions": len(translation_result.translated_solutions),
                "concept_mappings": sum(len(s.concept_mappings) for s in translation_result.translated_solutions),
                "feasibility_scores": [s.feasibility_score for s in translation_result.translated_solutions],
                "translation_confidence": translation_result.overall_confidence
            },
            confidence=translation_result.overall_confidence,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            metadata={
                "translation_strategy": translation_result.translation_strategy,
                "domain_vocabulary_size": len(translation_result.domain_vocabulary)
            }
        )
        
        self.reasoning_steps.append(reasoning_step)
        
        # Weave tracing captures this data automatically via @weave.op() decorator
        
        # Trace methodology transfers
        self._trace_methodology_transfers(translation_result.translated_solutions)
        
        # Record translation insights
        insights = self._extract_translation_insights(translation_result)
        self.key_insights.extend(insights)
        
        self.logger.info(f"Traced translation step: {step_id}")
    
    @weave.op()
    def record_decision(self, decision_id: str, description: str,
                       options: List[str], chosen_option: str,
                       reasoning: str, confidence: float,
                       impact: str) -> None:
        """
        Record a decision point in the reasoning process
        
        Args:
            decision_id: Unique identifier for the decision
            description: Description of what was being decided
            options: List of options considered
            chosen_option: The option that was selected
            reasoning: Explanation of why this option was chosen
            confidence: Confidence in the decision (0-1)
            impact: Description of how this impacts the overall process
        """
        decision = DecisionPoint(
            decision_id=decision_id,
            description=description,
            options_considered=options,
            chosen_option=chosen_option,
            reasoning=reasoning,
            confidence=confidence,
            impact_on_process=impact
        )
        
        self.decision_points.append(decision)
        
        # Weave tracing captures this data automatically via @weave.op() decorator
        
        self.logger.info(f"Recorded decision: {decision_id}")
    
    @weave.op()
    def add_insight(self, insight: str, source_step: Optional[str] = None) -> None:
        """
        Add a key insight discovered during the process
        
        Args:
            insight: The insight discovered
            source_step: Optional step ID where this insight was discovered
        """
        formatted_insight = insight
        if source_step:
            formatted_insight = f"[{source_step}] {insight}"
        
        self.key_insights.append(formatted_insight)
        
        # Weave tracing captures this data automatically via @weave.op() decorator
        
        self.logger.info(f"Added insight: {insight[:50]}...")
    
    @weave.op()
    def finalize_session(self) -> ProcessTrace:
        """
        Finalize the reasoning session and create complete trace
        
        Returns:
            Complete process trace for the session
        """
        end_time = datetime.now().isoformat()
        total_time = sum(step.execution_time for step in self.reasoning_steps)
        
        # Calculate success metrics
        success_metrics = self._calculate_success_metrics()
        
        # Create complete trace
        process_trace = ProcessTrace(
            session_id=self.session_id,
            original_problem_description=self.original_problem_description,
            start_time=self.start_time,
            end_time=end_time,
            reasoning_steps=self.reasoning_steps,
            decision_points=self.decision_points,
            similarity_justifications=self.similarity_justifications,
            key_insights=self.key_insights,
            methodology_transfers=self.methodology_transfers,
            success_metrics=success_metrics,
            total_execution_time=total_time
        )
        
        # Weave tracing captures this data automatically via @weave.op() decorator
        
        self.logger.info(f"Finalized reasoning session: {self.session_id}")
        return process_trace
    
    def _extract_abstraction_insights(self, abstracted_problem: AbstractedProblem) -> List[str]:
        """Extract insights from the abstraction process"""
        insights = []
        
        pattern = abstracted_problem.mathematical_pattern
        
        # Insights about mathematical complexity
        if len(pattern.mathematical_structures) > 2:
            insights.append(f"Problem involves multiple mathematical structures: {[s.value for s in pattern.mathematical_structures]}")
        
        # Insights about problem characteristics
        if pattern.iterative_process:
            insights.append("Problem involves iterative refinement, suggesting potential for active learning approaches")
        
        if pattern.uncertainty_sources:
            insights.append(f"Identified uncertainty sources: {pattern.uncertainty_sources}")
        
        # Insights about domain transferability
        if len(pattern.domain_agnostic_terms) > 5:
            insights.append("Rich set of domain-agnostic terms suggests high transferability potential")
        
        return insights
    
    def _extract_search_insights(self, search_result: SearchResult) -> List[str]:
        """Extract insights from the search process"""
        insights = []
        
        # Coverage insights
        if search_result.total_papers_analyzed > 100:
            insights.append("Comprehensive search coverage with extensive paper analysis")
        
        # Domain diversity insights
        unique_domains = set(m.domain for m in search_result.matches)
        if len(unique_domains) > 3:
            insights.append(f"Found solutions across {len(unique_domains)} different domains: {list(unique_domains)}")
        
        # Quality insights
        high_similarity_matches = [m for m in search_result.matches if m.similarity_score.overall_similarity > 0.7]
        if high_similarity_matches:
            insights.append(f"Found {len(high_similarity_matches)} high-similarity matches (>0.7)")
        
        # Impact insights
        high_impact_threshold = config_manager.get('temporal', 'citation_thresholds.high_impact', 50)
        high_impact_papers = [m for m in search_result.matches if m.citation_count > high_impact_threshold]
        if high_impact_papers:
            insights.append(f"Discovered {len(high_impact_papers)} high-impact papers with significant citations")
        
        return insights
    
    def _extract_translation_insights(self, translation_result: TranslationResult) -> List[str]:
        """Extract key insights from translation results"""
        
        insights = []
        
        if translation_result.translated_solutions:
            # Top solution insights
            top_solutions = translation_result.translated_solutions[:3]  # Top 3 solutions
            for solution in top_solutions:
                insights.append(f"High-feasibility solution from {solution.source_paper.domain}: "
                             f"{solution.methodology_translation.translated_method[:100]}...")
            
            # Cross-domain diversity
            unique_domains = set(s.source_paper.domain for s in translation_result.translated_solutions)
            if len(unique_domains) > 2:
                insights.append(f"Found solutions from {len(unique_domains)} different domains: {', '.join(unique_domains)}")
            
            # Translation quality
            avg_confidence = translation_result.overall_confidence
            insights.append(f"Translation confidence: {avg_confidence:.2f}")
        
        return insights
    
    def _trace_similarity_justifications(self, matches: List[CrossDomainMatch]) -> None:
        """Trace detailed similarity justifications for top matches"""
        
        for match in matches:
            reasoning = self._generate_similarity_reasoning(match)
            
            justification = SimilarityJustification(
                paper_id=match.paper_id,
                paper_title=match.title,
                overall_similarity=match.similarity_score.overall_similarity,
                component_scores={
                    "mathematical": match.similarity_score.mathematical_similarity,
                    "methodological": match.similarity_score.methodological_similarity,
                    "structural": match.similarity_score.structural_similarity
                },
                matching_evidence=match.matching_patterns,
                reasoning_chain=reasoning
            )
            
            self.similarity_justifications.append(justification)
    
    def _generate_similarity_reasoning(self, match: CrossDomainMatch) -> List[str]:
        """Generate reasoning chain for similarity score"""
        
        reasoning = []
        
        # Mathematical similarity reasoning
        if match.similarity_score.mathematical_similarity > 0.6:
            reasoning.append("Strong mathematical structure alignment")
        
        # Methodological similarity reasoning
        if match.similarity_score.methodological_similarity > 0.6:
            reasoning.append("High methodological approach similarity")
        
        # Pattern matching reasoning
        if match.matching_patterns:
            patterns = match.matching_patterns[:3]  # Top 3 patterns
            reasoning.append(f"Multiple pattern matches: {', '.join(patterns)}")
        
        # Citation impact reasoning
        if match.citation_count > 50:
            reasoning.append("High-impact paper with strong community validation")
        
        return reasoning
    
    def _trace_methodology_transfers(self, solutions: List[TranslatedSolution]) -> None:
        """Trace methodology transfer details"""
        
        for solution in solutions:
            transfer = {
                "source_paper": solution.source_paper.title,
                "source_domain": solution.source_paper.domain,
                "target_domain": solution.target_domain,
                "original_method": solution.methodology_translation.original_method,
                "translated_method": solution.methodology_translation.translated_method,
                "key_adaptations": solution.methodology_translation.adaptation_steps[:3],
                "feasibility_score": solution.feasibility_score,
                "confidence": solution.translation_confidence
            }
            
            self.methodology_transfers.append(transfer)
    
    def _calculate_success_metrics(self) -> Dict[str, float]:
        """Calculate success metrics for the entire process"""
        metrics = {}
        
        # Completion metrics
        metrics["steps_completed"] = len(self.reasoning_steps)
        metrics["decisions_made"] = len(self.decision_points)
        metrics["insights_generated"] = len(self.key_insights)
        
        # Quality metrics
        if self.reasoning_steps:
            avg_confidence = sum(step.confidence for step in self.reasoning_steps) / len(self.reasoning_steps)
            metrics["average_step_confidence"] = avg_confidence
        
        if self.decision_points:
            avg_decision_confidence = sum(dp.confidence for dp in self.decision_points) / len(self.decision_points)
            metrics["average_decision_confidence"] = avg_decision_confidence
        
        # Translation success metrics
        if self.methodology_transfers:
            feasible_transfers = sum(1 for t in self.methodology_transfers if t["feasibility_score"] > 0.6)
            metrics["feasible_transfer_ratio"] = feasible_transfers / len(self.methodology_transfers)
            
            avg_transfer_confidence = sum(t["confidence"] for t in self.methodology_transfers) / len(self.methodology_transfers)
            metrics["average_transfer_confidence"] = avg_transfer_confidence
        
        # Coverage metrics
        if self.similarity_justifications:
            high_similarity_ratio = sum(1 for sj in self.similarity_justifications if sj.overall_similarity > 0.6) / len(self.similarity_justifications)
            metrics["high_similarity_ratio"] = high_similarity_ratio
        
        return metrics
    
    @weave.op()
    def get_trace_summary(self) -> Dict[str, Any]:
        """Get a summary of the current reasoning trace"""
        return {
            "session_id": self.session_id,
            "steps_completed": len(self.reasoning_steps),
            "decision_points": len(self.decision_points),
            "insights_generated": len(self.key_insights),
            "similarity_justifications": len(self.similarity_justifications),
            "methodology_transfers": len(self.methodology_transfers),
            "current_step": self.current_step,
            "session_duration": self._calculate_session_duration()
        }
    
    def _calculate_session_duration(self) -> float:
        """Calculate current session duration in seconds"""
        start = datetime.fromisoformat(self.start_time)
        now = datetime.now()
        return (now - start).total_seconds()
    
    @weave.op()
    def export_trace_for_analysis(self) -> Dict[str, Any]:
        """Export complete trace data for external analysis"""
        return {
            "session_metadata": {
                "session_id": self.session_id,
                "start_time": self.start_time,
                "problem_description": self.original_problem_description
            },
            "reasoning_steps": [asdict(step) for step in self.reasoning_steps],
            "decision_points": [asdict(dp) for dp in self.decision_points],
            "similarity_justifications": [asdict(sj) for sj in self.similarity_justifications],
            "key_insights": self.key_insights,
            "methodology_transfers": self.methodology_transfers,
            "success_metrics": self._calculate_success_metrics()
        }