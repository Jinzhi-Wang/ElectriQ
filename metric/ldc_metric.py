"""
LDC (Long-Dialogue Consistency) Metric Implementation
LDC evaluates whether models maintain stable process reasoning and state
tracking over long multi-turn dialogues.

Formula: SLDC = 100 × (0.7 × SCS + 0.2 × STV + 0.1 × RCI)

Where:
- SCS: Slot Consistency Score (per-turn slot agreement or justified updates)
- STV: State Transition Validity (share of valid FSM transitions)
- RCI: Reference and Citation Integrity (continuation or justified updates of sources)
"""

from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np


class SlotUpdateType(Enum):
    """Types of slot updates in dialogue."""
    CONSISTENT = 1.0  # Slot value remains consistent
    JUSTIFIED_UPDATE = 1.0  # Evidence-backed update/refinement
    TEMPORARY_OMISSION = 0.5  # Temporary omission of committed value
    CONTRADICTION = 0.0  # Unjustified change/contradiction


class LDCMetric:
    """
    Long-Dialogue Consistency metric for evaluating multi-turn dialogue quality.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize LDC metric.

        Args:
            weights: Custom weights for SCS, STV, RCI (default: SCS=0.7, STV=0.2, RCI=0.1)
        """
        self.weights = weights if weights else {
            'SCS': 0.7,
            'STV': 0.2,
            'RCI': 0.1
        }

    def _extract_slots(self, response: str, slot_keys: List[str]) -> Dict[str, Any]:
        """
        Extract slot values from response.

        Args:
            response: Model response text
            slot_keys: List of slot keys to extract

        Returns:
            dict: Extracted slot values
        """
        import re
        slots = {}

        for key in slot_keys:
            # Generic pattern for extracting slot values
            patterns = [
                rf'{key}[:\s]+([^\n,\.]+)',
                rf'{key}\s*is\s*([^\n,\.]+)',
                rf'{key}\s*=\s*([^\n,\.]+)'
            ]

            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    slots[key] = match.group(1).strip()
                    break

            if key not in slots:
                slots[key] = None

        return slots

    def _compare_slot_values(self, prev_value: Any, curr_value: Any,
                             has_evidence: bool = False) -> float:
        """
        Compare slot values between turns.

        Args:
            prev_value: Previous turn slot value
            curr_value: Current turn slot value
            has_evidence: Whether update is evidence-backed

        Returns:
            float: Consistency score (0, 0.5, or 1)
        """
        # Both values are None or missing
        if prev_value is None and curr_value is None:
            return SlotUpdateType.CONSISTENT.value

        # Previous exists, current is omitted
        if prev_value is not None and curr_value is None:
            return SlotUpdateType.TEMPORARY_OMISSION.value

        # Previous is None, current has value (new information)
        if prev_value is None and curr_value is not None:
            return SlotUpdateType.CONSISTENT.value

        # Both have values - check consistency
        if str(prev_value).lower() == str(curr_value).lower():
            return SlotUpdateType.CONSISTENT.value

        # Values differ - check if justified
        if has_evidence:
            return SlotUpdateType.JUSTIFIED_UPDATE.value

        return SlotUpdateType.CONTRADICTION.value

    def compute_slot_consistency_score(self, dialogue_turns: List[Dict],
                                       slot_keys: List[str],
                                       slot_weights: Optional[Dict[str, float]] = None
                                       ) -> float:
        """
        Compute Slot Consistency Score (SCS).

        Args:
            dialogue_turns: List of dialogue turn dictionaries with 'response' and 'slots'
            slot_keys: List of slot keys to track
            slot_weights: Weights for each slot (default: uniform)

        Returns:
            float: SCS score (0-1)
        """
        if slot_weights is None:
            slot_weights = {key: 1.0 for key in slot_keys}

        if len(dialogue_turns) < 2:
            return 1.0  # Single turn dialogues are fully consistent

        total_phi = 0.0
        total_weight = 0.0

        for t in range(1, len(dialogue_turns)):
            prev_turn = dialogue_turns[t - 1]
            curr_turn = dialogue_turns[t]

            prev_slots = prev_turn.get('slots', {})
            curr_slots = curr_turn.get('slots', {})
            has_evidence = curr_turn.get('has_evidence', False)

            for key in slot_keys:
                weight = slot_weights.get(key, 1.0)
                phi = self._compare_slot_values(
                    prev_slots.get(key),
                    curr_slots.get(key),
                    has_evidence
                )
                total_phi += weight * phi
                total_weight += weight

        return total_phi / total_weight if total_weight > 0 else 0.0

    def compute_state_transition_validity(self, dialogue_turns: List[Dict],
                                          valid_transitions: List[Tuple[str, str]]
                                          ) -> float:
        """
        Compute State Transition Validity (STV) score.

        Args:
            dialogue_turns: List of dialogue turns with 'state' field
            valid_transitions: List of valid state transition pairs

        Returns:
            float: STV score (0-1)
        """
        if len(dialogue_turns) < 2:
            return 1.0

        valid_count = 0
        total_transitions = len(dialogue_turns) - 1

        for t in range(1, len(dialogue_turns)):
            prev_state = dialogue_turns[t - 1].get('state', '')
            curr_state = dialogue_turns[t].get('state', '')

            transition = (prev_state, curr_state)
            if transition in valid_transitions:
                valid_count += 1

        return valid_count / total_transitions if total_transitions > 0 else 0.0

    def compute_reference_citation_integrity(self, dialogue_turns: List[Dict]
                                             ) -> float:
        """
        Compute Reference and Citation Integrity (RCI) score.

        Args:
            dialogue_turns: List of dialogue turns with 'citation' field

        Returns:
            float: RCI score (0-1)
        """
        if len(dialogue_turns) == 0:
            return 0.0

        integrity_count = 0

        for t, turn in enumerate(dialogue_turns):
            citation = turn.get('citation', None)
            justified_update = turn.get('justified_citation_update', False)

            # First turn with citation establishes the source
            if t == 0:
                if citation is not None:
                    integrity_count += 1
            else:
                prev_citation = dialogue_turns[t - 1].get('citation', None)

                # Continue using existing source
                if citation == prev_citation and citation is not None:
                    integrity_count += 1
                # Justified update of source
                elif justified_update:
                    integrity_count += 1
                # No citation required (informational turn)
                elif citation is None and prev_citation is None:
                    integrity_count += 1

        return integrity_count / len(dialogue_turns)

    def compute_ldc(self, dialogue_turns: List[Dict],
                    slot_keys: List[str],
                    valid_transitions: Optional[List[Tuple[str, str]]] = None,
                    slot_weights: Optional[Dict[str, float]] = None
                    ) -> Dict:
        """
        Compute LDC score for a multi-turn dialogue.

        Args:
            dialogue_turns: List of dialogue turn dictionaries
            slot_keys: List of slot keys to track
            valid_transitions: List of valid state transitions
            slot_weights: Weights for each slot

        Returns:
            dict: LDC score and components
        """
        # Compute component scores
        SCS = self.compute_slot_consistency_score(
            dialogue_turns, slot_keys, slot_weights
        )

        STV = 0.0
        if valid_transitions:
            STV = self.compute_state_transition_validity(
                dialogue_turns, valid_transitions
            )
        else:
            STV = 1.0  # Default to 1 if no FSM defined

        RCI = self.compute_reference_citation_integrity(dialogue_turns)

        # Compute weighted LDC score
        ldc_score = 100 * (
                self.weights['SCS'] * SCS +
                self.weights['STV'] * STV +
                self.weights['RCI'] * RCI
        )

        return {
            'LDC': ldc_score,
            'SCS': SCS,
            'STV': STV,
            'RCI': RCI
        }

    def compute_ldc_batch(self, dialogues: List[List[Dict]],
                          slot_keys: List[str],
                          valid_transitions: Optional[List[Tuple[str, str]]] = None,
                          slot_weights: Optional[Dict[str, float]] = None
                          ) -> Dict:
        """
        Compute LDC scores for multiple dialogues.

        Args:
            dialogues: List of dialogues (each dialogue is a list of turns)
            slot_keys: List of slot keys to track
            valid_transitions: List of valid state transitions
            slot_weights: Weights for each slot

        Returns:
            dict: Aggregate LDC statistics
        """
        ldc_scores = []
        scs_scores, stv_scores, rci_scores = [], [], []

        for dialogue in dialogues:
            result = self.compute_ldc(
                dialogue, slot_keys, valid_transitions, slot_weights
            )
            ldc_scores.append(result['LDC'])
            scs_scores.append(result['SCS'])
            stv_scores.append(result['STV'])
            rci_scores.append(result['RCI'])

        return {
            'LDC': {
                'average': np.mean(ldc_scores),
                'std': np.std(ldc_scores),
                'min': np.min(ldc_scores),
                'max': np.max(ldc_scores),
                'scores': ldc_scores
            },
            'SCS': {
                'average': np.mean(scs_scores),
                'std': np.std(scs_scores)
            },
            'STV': {
                'average': np.mean(stv_scores),
                'std': np.std(stv_scores)
            },
            'RCI': {
                'average': np.mean(rci_scores),
                'std': np.std(rci_scores)
            }
        }


def evaluate_ldc(dialogues: List[List[Dict]],
                 slot_keys: List[str],
                 valid_transitions: Optional[List[Tuple[str, str]]] = None) -> Dict:
    """
    Convenience function to evaluate LDC scores.

    Args:
        dialogues: List of dialogues (each dialogue is a list of turns)
        slot_keys: List of slot keys to track
        valid_transitions: List of valid state transitions

    Returns:
        dict: LDC evaluation results
    """
    metric = LDCMetric()
    return metric.compute_ldc_batch(dialogues, slot_keys, valid_transitions)


if __name__ == '__main__':
    # Example usage
    # Simulated 3-turn dialogue
    dialogue = [
        {
            'response': 'Off-peak period is from 23:00 to 07:00. Apply for Plan A through the app.',
            'slots': {
                'valley_hours': '23:00-07:00',
                'plan': 'A',
                'channel': 'app',
                'obj': 'residential'
            },
            'state': 'initial',
            'citation': 'Tariff Directory 2024, Article 12',
            'has_evidence': False,
            'justified_citation_update': False
        },
        {
            'response': 'In the new district, off-peak is 22:30-07:00 per Article 9 of 2024-08 New District Tariff.',
            'slots': {
                'valley_hours': '22:30-07:00',
                'plan': 'A',
                'channel': 'app',
                'obj': 'residential'
            },
            'state': 'updated',
            'citation': '2024-08 New District Tariff Directory, Article 9',
            'has_evidence': True,
            'justified_citation_update': True
        },
        {
            'response': 'Submit Plan A application through Tariff Services section in the app.',
            'slots': {
                'valley_hours': None,
                'plan': 'A',
                'channel': 'app',
                'obj': None
            },
            'state': 'final',
            'citation': '2024-08 New District Tariff Directory, Article 9',
            'has_evidence': False,
            'justified_citation_update': False
        }
    ]

    dialogues = [dialogue]
    slot_keys = ['valley_hours', 'plan', 'channel', 'obj']

    valid_transitions = [
        ('initial', 'updated'),
        ('updated', 'final'),
        ('initial', 'final')
    ]

    results = evaluate_ldc(dialogues, slot_keys, valid_transitions)
    print(f"LDC Average Score: {results['LDC']['average']:.2f}")
    print(f"Slot Consistency (SCS): {results['SCS']['average']:.3f}")
    print(f"State Transition Validity (STV): {results['STV']['average']:.3f}")
    print(f"Reference Citation Integrity (RCI): {results['RCI']['average']:.3f}")