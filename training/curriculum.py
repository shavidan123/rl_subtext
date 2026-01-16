"""
Curriculum learning scheduler for the RL Subtext Transfer experiment.

Manages the progression of difficulty through training:
- Phase 1: Single payload, long token limit, heavy shaping
- Phase 2: Single payload, shorter limit, medium shaping
- Phase 3: Multiple payloads, short limit, sparse rewards
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CurriculumPhase:
    """Configuration for a curriculum phase."""
    name: str
    start_episode: int
    end_episode: int
    payloads: List[str]
    carrier_token_limit: int
    reward_shaping: str


class CurriculumScheduler:
    """
    Manages curriculum progression through training phases.
    """

    def __init__(self, phases: List[CurriculumPhase]):
        """
        Initialize scheduler with phases.

        Args:
            phases: List of curriculum phases in order
        """
        self.phases = sorted(phases, key=lambda p: p.start_episode)
        self.current_phase_idx = 0
        self.current_episode = 0

    @classmethod
    def from_config(cls, config: Dict) -> "CurriculumScheduler":
        """
        Create scheduler from config dict.

        Args:
            config: Experiment configuration

        Returns:
            Configured CurriculumScheduler
        """
        curriculum_config = config.get("training", {}).get("curriculum", {})

        if not curriculum_config.get("enabled", False):
            # Return a single-phase scheduler with defaults
            return cls([
                CurriculumPhase(
                    name="default",
                    start_episode=0,
                    end_episode=float("inf"),
                    payloads=["pirate"],
                    carrier_token_limit=200,
                    reward_shaping="sparse",
                )
            ])

        phases = []
        for phase_dict in curriculum_config.get("phases", []):
            phases.append(CurriculumPhase(
                name=phase_dict.get("name", "unnamed"),
                start_episode=phase_dict.get("start_episode", 0),
                end_episode=phase_dict.get("end_episode", float("inf")),
                payloads=phase_dict.get("payloads", ["pirate"]),
                carrier_token_limit=phase_dict.get("carrier_token_limit", 200),
                reward_shaping=phase_dict.get("reward_shaping", "sparse"),
            ))

        return cls(phases)

    def step(self, episode: int) -> Optional[CurriculumPhase]:
        """
        Update to the given episode and return current phase if changed.

        Args:
            episode: Current episode number

        Returns:
            New CurriculumPhase if phase changed, None otherwise
        """
        self.current_episode = episode

        # Find the appropriate phase
        new_phase_idx = self.current_phase_idx

        for i, phase in enumerate(self.phases):
            if phase.start_episode <= episode < phase.end_episode:
                new_phase_idx = i
                break

        # Check if phase changed
        if new_phase_idx != self.current_phase_idx:
            self.current_phase_idx = new_phase_idx
            return self.current_phase

        return None

    @property
    def current_phase(self) -> CurriculumPhase:
        """Get current curriculum phase."""
        return self.phases[self.current_phase_idx]

    @property
    def payloads(self) -> List[str]:
        """Get current payload names."""
        return self.current_phase.payloads

    @property
    def token_limit(self) -> int:
        """Get current token limit."""
        return self.current_phase.carrier_token_limit

    @property
    def reward_shaping(self) -> str:
        """Get current reward shaping mode."""
        return self.current_phase.reward_shaping

    def get_progress(self) -> Dict:
        """Get curriculum progress information."""
        phase = self.current_phase
        phase_progress = 0.0

        if phase.end_episode != float("inf"):
            phase_duration = phase.end_episode - phase.start_episode
            phase_progress = (self.current_episode - phase.start_episode) / phase_duration
            phase_progress = min(1.0, max(0.0, phase_progress))

        return {
            "episode": self.current_episode,
            "phase_name": phase.name,
            "phase_idx": self.current_phase_idx,
            "total_phases": len(self.phases),
            "phase_progress": phase_progress,
            "payloads": phase.payloads,
            "token_limit": phase.carrier_token_limit,
            "reward_shaping": phase.reward_shaping,
        }

    def is_complete(self) -> bool:
        """Check if curriculum is complete."""
        if not self.phases:
            return True

        last_phase = self.phases[-1]
        return self.current_episode >= last_phase.end_episode
