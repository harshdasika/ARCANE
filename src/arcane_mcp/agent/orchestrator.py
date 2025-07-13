"""
Cross-Domain Research Agent Orchestrator - Main agent workflow coordination
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import weave

from .problem_abstractor import ProblemAbstractor, AbstractedProblem
from .cross_domain_searcher import CrossDomainSearcher, SearchResult
from .solution_translator import SolutionTranslator, TranslationResult
from .reasoning_tracer import ReasoningTracer, ProcessTrace

logger = logging.getLogger(__name__)

@dataclass
class AgentResult:
    """Complete result from the cross-domain research agent"""
    session_id: str
    original_problem: str
    abstracted_problem: AbstractedProblem
    search_result: SearchResult
    translation_result: TranslationResult
    process_trace: ProcessTrace
    execution_summary: Dict[str, Any]
    recommendations: List[str]

class CrossDomainResearchAgent:
    """Main orchestrator for cross-domain research discovery"""
    
    def __init__(self, arcane_clients: Dict[str, Any]):
        """
        Initialize the cross-domain research agent
        
        Args:
            arcane_clients: Dictionary containing ARCANE API clients and components
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.problem_abstractor = ProblemAbstractor()
        self.cross_domain_searcher = CrossDomainSearcher(arcane_clients)
        self.solution_translator = SolutionTranslator()
        
        # ARCANE clients for integration
        self.arcane_clients = arcane_clients
        
        self.logger.info("Cross-domain research agent initialized")
    
    @weave.op()
    async def analyze_research_problem(self, problem_description: str,
                                     max_solutions: int = 10,
                                     session_id: Optional[str] = None) -> AgentResult:
        """
        Main method to analyze a research problem and discover cross-domain solutions
        
        Args:
            problem_description: Natural language description of the research problem
            max_solutions: Maximum number of cross-domain solutions to return
            session_id: Optional session ID for tracking
            
        Returns:
            AgentResult containing complete analysis and recommendations
        """
        try:
            self.logger.info(f"Starting cross-domain analysis for problem: {problem_description[:100]}...")
            
            # Initialize reasoning tracer
            tracer = ReasoningTracer(session_id)
            tracer.start_session(problem_description)
            
            # Phase 1: Problem Abstraction
            self.logger.info("Phase 1: Abstracting research problem")
            start_time = time.time()
            
            abstracted_problem = await self.problem_abstractor.abstract_problem(problem_description)
            
            abstraction_time = time.time() - start_time
            tracer.trace_abstraction_step(problem_description, abstracted_problem, abstraction_time)
            
            # Record abstraction decisions
            await self._record_abstraction_decisions(tracer, abstracted_problem)
            
            # Phase 2: Cross-Domain Search
            self.logger.info("Phase 2: Searching across domains")
            start_time = time.time()
            
            search_result = await self.cross_domain_searcher.search_cross_domain(
                abstracted_problem, max_results=max_solutions * 2  # Search for more to filter better
            )
            
            search_time = time.time() - start_time
            tracer.trace_search_step(abstracted_problem, search_result, search_time)
            
            # Record search decisions
            await self._record_search_decisions(tracer, search_result)
            
            # Phase 3: Solution Translation
            self.logger.info("Phase 3: Translating solutions to target domain")
            start_time = time.time()
            
            translation_result = await self.solution_translator.translate_solutions(search_result)
            
            translation_time = time.time() - start_time
            tracer.trace_translation_step(search_result, translation_result, translation_time)
            
            # Record translation decisions
            await self._record_translation_decisions(tracer, translation_result)
            
            # Phase 4: Generate Recommendations
            self.logger.info("Phase 4: Generating recommendations")
            recommendations = await self._generate_recommendations(
                abstracted_problem, search_result, translation_result, tracer
            )
            
            # Finalize reasoning trace
            process_trace = tracer.finalize_session()
            
            # Create execution summary
            execution_summary = self._create_execution_summary(
                abstracted_problem, search_result, translation_result, process_trace
            )
            
            # Create final result
            agent_result = AgentResult(
                session_id=tracer.session_id,
                original_problem=problem_description,
                abstracted_problem=abstracted_problem,
                search_result=search_result,
                translation_result=translation_result,
                process_trace=process_trace,
                execution_summary=execution_summary,
                recommendations=recommendations
            )
            
            self.logger.info(f"Cross-domain analysis completed successfully: session {tracer.session_id}")
            return agent_result
            
        except Exception as e:
            self.logger.error(f"Error in cross-domain analysis: {str(e)}")
            raise
    
    @weave.op()
    async def _record_abstraction_decisions(self, tracer: ReasoningTracer, 
                                          abstracted_problem: AbstractedProblem) -> None:
        """Record decisions made during problem abstraction"""
        
        pattern = abstracted_problem.mathematical_pattern
        
        # Decision: Problem type classification
        if pattern.problem_types:
            problem_types = [pt.value for pt in pattern.problem_types]
            tracer.record_decision(
                decision_id="problem_type_classification",
                description="Classify the primary problem types",
                options=["optimization", "prediction", "classification", "uncertainty_quantification", "active_learning"],
                chosen_option=", ".join(problem_types),
                reasoning=f"Based on keyword analysis and mathematical structure detection, identified: {problem_types}",
                confidence=pattern.confidence_score,
                impact="Determines search strategy and similarity matching criteria"
            )
        
        # Decision: Mathematical structure identification
        if pattern.mathematical_structures:
            structures = [ms.value for ms in pattern.mathematical_structures]
            tracer.record_decision(
                decision_id="mathematical_structure_identification",
                description="Identify core mathematical structures",
                options=["optimization_theory", "probability_theory", "linear_algebra", "graph_theory"],
                chosen_option=", ".join(structures),
                reasoning=f"Mathematical analysis revealed these core structures: {structures}",
                confidence=pattern.confidence_score,
                impact="Guides cross-domain similarity assessment and method translation"
            )
        
        # Decision: Domain-agnostic term extraction
        if pattern.domain_agnostic_terms:
            tracer.record_decision(
                decision_id="domain_agnostic_terms",
                description="Extract domain-agnostic search terms",
                options=pattern.domain_agnostic_terms + ["generic terms", "domain-specific terms only"],
                chosen_option=f"Selected {len(pattern.domain_agnostic_terms)} terms",
                reasoning="Chose terms that abstract away domain-specific vocabulary while preserving methodological essence",
                confidence=0.8,
                impact="Enables effective cross-domain search by focusing on transferable concepts"
            )
    
    @weave.op()
    async def _record_search_decisions(self, tracer: ReasoningTracer, 
                                     search_result: SearchResult) -> None:
        """Record decisions made during cross-domain search"""
        
        # Decision: Domain selection for search
        domains_searched = search_result.domains_searched
        tracer.record_decision(
            decision_id="domain_selection",
            description="Select target domains for cross-domain search",
            options=["all_domains", "high_relevance_only", "similar_domains_only"],
            chosen_option=f"Selected {len(domains_searched)} domains: {', '.join(domains_searched)}",
            reasoning="Chose domains based on mathematical pattern relevance and potential for transferable solutions",
            confidence=search_result.search_confidence,
            impact="Determines breadth vs depth of search and quality of cross-domain matches"
        )
        
        # Decision: Similarity threshold
        if search_result.matches:
            min_similarity = min(m.similarity_score.overall_similarity for m in search_result.matches)
            max_similarity = max(m.similarity_score.overall_similarity for m in search_result.matches)
            tracer.record_decision(
                decision_id="similarity_threshold",
                description="Set minimum similarity threshold for paper matching",
                options=["0.3 (permissive)", "0.5 (moderate)", "0.7 (strict)"],
                chosen_option=f"Used dynamic threshold, accepted papers with similarity {min_similarity:.2f}-{max_similarity:.2f}",
                reasoning="Balanced between finding diverse solutions and maintaining methodological relevance",
                confidence=0.7,
                impact="Affects quality vs quantity of cross-domain matches"
            )
        
        # Decision: Paper ranking strategy
        high_similarity_papers = [m for m in search_result.matches if m.similarity_score.overall_similarity > 0.6]
        tracer.record_decision(
            decision_id="paper_ranking",
            description="Strategy for ranking and selecting papers",
            options=["similarity_only", "impact_weighted", "diversity_balanced"],
            chosen_option="similarity_weighted with impact consideration",
            reasoning=f"Prioritized {len(high_similarity_papers)} high-similarity papers while considering citation impact",
            confidence=0.8,
            impact="Determines which solutions get translated and recommended"
        )
    
    @weave.op()
    async def _record_translation_decisions(self, tracer: ReasoningTracer,
                                          translation_result: TranslationResult) -> None:
        """Record decisions made during solution translation"""
        
        # Decision: Concept mapping strategy
        total_mappings = sum(len(s.concept_mappings) for s in translation_result.translated_solutions)
        tracer.record_decision(
            decision_id="concept_mapping_strategy",
            description="Strategy for mapping concepts between domains",
            options=["direct_mapping", "semantic_similarity", "contextual_adaptation"],
            chosen_option="Multi-level mapping with confidence weighting",
            reasoning=f"Generated {total_mappings} concept mappings using predefined rules + contextual analysis",
            confidence=translation_result.overall_confidence,
            impact="Determines accuracy and usefulness of translated solutions"
        )
        
        # Decision: Feasibility assessment
        feasible_solutions = [s for s in translation_result.translated_solutions if s.feasibility_score > 0.6]
        tracer.record_decision(
            decision_id="feasibility_assessment",
            description="Assess feasibility of translated solutions",
            options=["optimistic", "realistic", "conservative"],
            chosen_option="Realistic assessment with multiple criteria",
            reasoning=f"Identified {len(feasible_solutions)} feasible solutions using confidence, mapping quality, and domain compatibility",
            confidence=0.75,
            impact="Guides implementation recommendations and priority ranking"
        )
        
        # Decision: Implementation roadmap depth
        tracer.record_decision(
            decision_id="implementation_detail",
            description="Level of detail for implementation roadmaps",
            options=["high_level_only", "detailed_steps", "comprehensive_guidance"],
            chosen_option="Structured roadmap with phase-based implementation",
            reasoning="Provided actionable roadmaps balancing detail with practical applicability",
            confidence=0.7,
            impact="Affects practical utility for researchers implementing solutions"
        )
    
    @weave.op()
    async def _generate_recommendations(self, abstracted_problem: AbstractedProblem,
                                      search_result: SearchResult,
                                      translation_result: TranslationResult,
                                      tracer: ReasoningTracer) -> List[str]:
        """Generate actionable recommendations based on the analysis"""
        
        recommendations = []
        
        # Top solution recommendations
        if translation_result.translated_solutions:
            top_solutions = sorted(translation_result.translated_solutions, 
                                 key=lambda s: s.feasibility_score, reverse=True)[:3]
            
            for i, solution in enumerate(top_solutions, 1):
                source_domain = solution.source_paper.domain
                target_domain = solution.target_domain
                recommendations.append(
                    f"#{i} Priority Solution: Adapt methodology from {source_domain} paper "
                    f"'{solution.source_paper.title[:60]}...' (feasibility: {solution.feasibility_score:.2f}). "
                    f"Key adaptation: {solution.methodology_translation.translated_method[:100]}..."
                )
        
        # Methodological insights
        if abstracted_problem.mathematical_pattern.iterative_process:
            recommendations.append(
                "Consider iterative/adaptive approaches as your problem shows iterative characteristics. "
                "Several discovered solutions use active learning or adaptive sampling strategies."
            )
        
        if abstracted_problem.mathematical_pattern.uncertainty_sources:
            recommendations.append(
                "Incorporate uncertainty quantification methods as your problem involves uncertain elements. "
                "Found solutions address uncertainty through probabilistic modeling and robust optimization."
            )
        
        # Cross-domain insights
        if translation_result.translated_solutions:
            unique_domains = set(s.source_paper.domain for s in translation_result.translated_solutions)
            if len(unique_domains) > 2:
                recommendations.append(
                    f"Explore methodologies from {len(unique_domains)} different domains: {', '.join(unique_domains)}. "
                    "This diversity suggests multiple valid approaches to your problem structure."
                )
        
        # Implementation strategy recommendations
        if translation_result.translated_solutions:
            high_feasibility = [s for s in translation_result.translated_solutions if s.feasibility_score > 0.7]
            if high_feasibility:
                recommendations.append(
                    f"Start with {len(high_feasibility)} high-feasibility approaches for quicker implementation and validation. "
                    "These have strong concept mappings and clear adaptation paths."
                )
        
        # Research gap recommendations
        if search_result.search_confidence > 0.8 and len(translation_result.translated_solutions) < 3:
            recommendations.append(
                "High search confidence with limited solutions suggests a potential research gap. "
                "Consider developing novel approaches or hybrid methods combining discovered techniques."
            )
        
        # Validation recommendations
        recommendations.append(
            f"Validate adapted methods using {abstracted_problem.domain_context.primary_domain}-specific benchmarks. "
            "Compare against existing domain methods to demonstrate cross-domain transfer value."
        )
        
        # Collaboration recommendations
        if translation_result.translated_solutions:
            source_domains = [s.source_paper.domain for s in translation_result.translated_solutions[:3]]
            recommendations.append(
                f"Consider collaborating with researchers in {', '.join(set(source_domains))} "
                "to better understand implementation nuances and validate adaptations."
            )
        
        # Add insights as recommendations
        if hasattr(tracer, 'key_insights') and tracer.key_insights:
            key_insights = tracer.key_insights[-3:]  # Last 3 insights
            for insight in key_insights:
                if "insight" in insight.lower():
                    recommendations.append(f"Research Insight: {insight}")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _create_execution_summary(self, abstracted_problem: AbstractedProblem,
                                search_result: SearchResult,
                                translation_result: TranslationResult,
                                process_trace: ProcessTrace) -> Dict[str, Any]:
        """Create a comprehensive execution summary"""
        
        summary = {
            # Problem analysis summary
            "problem_analysis": {
                "domain": abstracted_problem.domain_context.primary_domain,
                "problem_types": [pt.value for pt in abstracted_problem.mathematical_pattern.problem_types],
                "mathematical_structures": [ms.value for ms in abstracted_problem.mathematical_pattern.mathematical_structures],
                "abstraction_confidence": abstracted_problem.abstraction_confidence,
                "iterative_process": abstracted_problem.mathematical_pattern.iterative_process,
                "uncertainty_involved": len(abstracted_problem.mathematical_pattern.uncertainty_sources) > 0
            },
            
            # Search results summary
            "search_results": {
                "domains_searched": search_result.domains_searched,
                "papers_analyzed": search_result.total_papers_analyzed,
                "matches_found": len(search_result.matches),
                "search_confidence": search_result.search_confidence,
                "top_similarity_score": max([m.similarity_score.overall_similarity for m in search_result.matches]) if search_result.matches else 0,
                "unique_source_domains": len(set(m.domain for m in search_result.matches))
            },
            
            # Translation summary
            "translation_results": {
                "solutions_translated": len(translation_result.translated_solutions),
                "concept_mappings_generated": sum(len(s.concept_mappings) for s in translation_result.translated_solutions),
                "feasible_solutions": len([s for s in translation_result.translated_solutions if s.feasibility_score > 0.6]),
                "translation_confidence": translation_result.overall_confidence,
                "average_feasibility": sum(s.feasibility_score for s in translation_result.translated_solutions) / len(translation_result.translated_solutions) if translation_result.translated_solutions else 0
            },
            
            # Process efficiency summary
            "process_metrics": {
                "total_execution_time": process_trace.total_execution_time,
                "reasoning_steps": len(process_trace.reasoning_steps),
                "decision_points": len(process_trace.decision_points),
                "insights_generated": len(process_trace.key_insights),
                "success_metrics": process_trace.success_metrics
            },
            
            # Quality indicators
            "quality_indicators": {
                "high_similarity_matches": len([m for m in search_result.matches if m.similarity_score.overall_similarity > 0.7]),
                "high_impact_papers": len([m for m in search_result.matches if m.citation_count > 50]),
                "recent_papers": len([m for m in search_result.matches if m.year and m.year > 2020]),
                "cross_domain_coverage": len(set(m.domain for m in search_result.matches)),
                "implementation_ready": len([s for s in translation_result.translated_solutions if s.feasibility_score > 0.8])
            }
        }
        
        return summary
    
    @weave.op()
    async def get_component_status(self) -> Dict[str, str]:
        """Get status of all agent components"""
        return {
            "problem_abstractor": "ready",
            "cross_domain_searcher": "ready",
            "solution_translator": "ready",
            "arcane_integration": "connected" if self.arcane_clients else "disconnected"
        }
    
    @weave.op()
    async def quick_analysis(self, problem_description: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Perform a quick analysis with limited depth for faster results
        
        Args:
            problem_description: Research problem description
            max_results: Maximum number of results (lower for speed)
            
        Returns:
            Simplified analysis results
        """
        try:
            self.logger.info("Starting quick cross-domain analysis")
            
            # Quick abstraction
            abstracted_problem = await self.problem_abstractor.abstract_problem(problem_description)
            
            # Limited search
            search_result = await self.cross_domain_searcher.search_cross_domain(
                abstracted_problem, max_results=max_results
            )
            
            # Quick translation for top results only
            if search_result.matches:
                limited_matches = search_result.matches[:max_results]
                limited_search = SearchResult(
                    original_problem=search_result.original_problem,
                    matches=limited_matches,
                    search_strategy=search_result.search_strategy,
                    domains_searched=search_result.domains_searched,
                    total_papers_analyzed=search_result.total_papers_analyzed,
                    search_confidence=search_result.search_confidence
                )
                
                translation_result = await self.solution_translator.translate_solutions(limited_search)
            else:
                translation_result = None
            
            # Quick recommendations
            quick_recommendations = []
            if translation_result and translation_result.translated_solutions:
                top_solution = max(translation_result.translated_solutions, key=lambda s: s.feasibility_score)
                quick_recommendations.append(
                    f"Top recommendation: Adapt '{top_solution.source_paper.title[:50]}...' "
                    f"from {top_solution.source_paper.domain} (feasibility: {top_solution.feasibility_score:.2f})"
                )
            
            return {
                "problem_domain": abstracted_problem.domain_context.primary_domain,
                "solutions_found": len(translation_result.translated_solutions) if translation_result else 0,
                "top_similarity": max([m.similarity_score.overall_similarity for m in search_result.matches]) if search_result.matches else 0,
                "feasible_solutions": len([s for s in translation_result.translated_solutions if s.feasibility_score > 0.6]) if translation_result else 0,
                "source_domains": list(set(s.source_paper.domain for s in translation_result.translated_solutions)) if translation_result else [],
                "quick_recommendations": quick_recommendations,
                "full_analysis_available": True
            }
            
        except Exception as e:
            self.logger.error(f"Error in quick analysis: {str(e)}")
            return {"error": str(e), "quick_recommendations": ["Unable to complete quick analysis"]}
    
    @weave.op()
    async def explain_methodology(self, solution_index: int, agent_result: AgentResult) -> Dict[str, Any]:
        """
        Provide detailed explanation of a specific translated methodology
        
        Args:
            solution_index: Index of the solution to explain
            agent_result: Previous agent result containing the solutions
            
        Returns:
            Detailed methodology explanation
        """
        try:
            if solution_index >= len(agent_result.translation_result.translated_solutions):
                return {"error": "Solution index out of range"}
            
            solution = agent_result.translation_result.translated_solutions[solution_index]
            
            explanation = {
                "source_paper": {
                    "title": solution.source_paper.title,
                    "domain": solution.source_paper.domain,
                    "authors": solution.source_paper.authors,
                    "year": solution.source_paper.year,
                    "citation_count": solution.source_paper.citation_count
                },
                "methodology_translation": {
                    "original_approach": solution.methodology_translation.original_method,
                    "adapted_approach": solution.methodology_translation.translated_method,
                    "adaptation_steps": solution.methodology_translation.adaptation_steps,
                    "implementation_notes": solution.methodology_translation.implementation_notes
                },
                "concept_mappings": [
                    {
                        "source": mapping.source_concept,
                        "target": mapping.target_concept,
                        "confidence": mapping.confidence,
                        "explanation": mapping.context
                    }
                    for mapping in solution.concept_mappings
                ],
                "mathematical_adaptations": solution.mathematical_adaptations,
                "implementation_roadmap": solution.implementation_roadmap,
                "feasibility_assessment": {
                    "score": solution.feasibility_score,
                    "confidence": solution.translation_confidence,
                    "expected_benefits": solution.expected_benefits,
                    "potential_challenges": solution.potential_limitations
                },
                "similarity_analysis": {
                    "overall_similarity": solution.source_paper.similarity_score.overall_similarity,
                    "mathematical_similarity": solution.source_paper.similarity_score.mathematical_similarity,
                    "methodological_similarity": solution.source_paper.similarity_score.methodological_similarity,
                    "matching_patterns": solution.source_paper.matching_patterns,
                    "relevance_explanation": solution.source_paper.relevance_explanation
                }
            }
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error explaining methodology: {str(e)}")
            return {"error": str(e)}