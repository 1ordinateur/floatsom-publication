"""
Simplified harmonization for merging best trials from multiple seed runs.

Uses study.best_trials to get the best parameter sets from each seed run,
then manually calculates the Pareto front from the combined results.
"""

from typing import List
import optuna
from optuna.trial import FrozenTrial

from .pareto_utils import validate_study_for_harmonization


class ParetoHarmonizer:
    """Simplified harmonizer for combining best trials from multiple seed runs."""
    
    def __init__(self):
        pass
        
    def harmonize_fronts(self, seed_studies: List[optuna.Study]) -> List[FrozenTrial]:
        """
        Extract best trials from each seed study and compute combined Pareto front.
        
        Args:
            seed_studies: List of studies from different random seeds
            
        Returns:
            List of Pareto-optimal trials from combined best trials
        """
        if not seed_studies:
            return []
            
        # Validate compatibility
        if not validate_study_for_harmonization(seed_studies):
            raise ValueError("Studies are not compatible for harmonization")
        
        # Extract best trials from each seed using study.best_trials
        all_best_trials = []
        for study in seed_studies:
            best_trials = study.best_trials
            if best_trials:
                all_best_trials.extend(best_trials)
        
        if not all_best_trials:
            return []
        
        # Manually calculate Pareto front from combined best trials
        return self._calculate_pareto_front(all_best_trials, seed_studies[0].directions)
    
    def _calculate_pareto_front(self, trials: List[FrozenTrial], 
                               directions: List[optuna.study.StudyDirection]) -> List[FrozenTrial]:
        """Calculate Pareto front from list of trials."""
        
        if not trials:
            return []
        
        if len(directions) == 1:
            # Single objective: return best trial
            if directions[0] == optuna.study.StudyDirection.MINIMIZE:
                return [min(trials, key=lambda t: t.value)]
            else:
                return [max(trials, key=lambda t: t.value)]
        
        # Multi-objective: find non-dominated solutions
        pareto_trials = []
        
        for candidate in trials:
            is_dominated = False
            
            for other in trials:
                if candidate == other:
                    continue
                
                # Check if candidate is dominated by other
                dominates = True
                strictly_better = False
                
                for i, direction in enumerate(directions):
                    candidate_val = candidate.values[i]
                    other_val = other.values[i]
                    
                    if direction == optuna.study.StudyDirection.MINIMIZE:
                        if other_val > candidate_val:  # Other is worse
                            dominates = False
                            break
                        elif other_val < candidate_val:  # Other is better
                            strictly_better = True
                    else:  # MAXIMIZE
                        if other_val < candidate_val:  # Other is worse
                            dominates = False
                            break
                        elif other_val > candidate_val:  # Other is better
                            strictly_better = True
                
                if dominates and strictly_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_trials.append(candidate)
        
        return pareto_trials