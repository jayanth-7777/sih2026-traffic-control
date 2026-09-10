from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import time

import config


class SignalPhase(Enum):
    NS_GREEN = "NORTH_SOUTH_GREEN"
    NS_AMBER = "NORTH_SOUTH_AMBER"
    EW_GREEN = "EAST_WEST_GREEN"
    EW_AMBER = "EAST_WEST_AMBER"


@dataclass
class DecisionEvent:
    timestamp: float
    rule: str
    phase: str
    action: str
    reason: str
    badge_type: str  # "success", "warning", "danger", "info"


class AdaptiveSignalController:
    """
    Step 4: Adaptive Signal Control Decision Engine
    Deterministic Python State Machine converting trajectory awareness into
    responsive traffic signal decisions.

    Rule 1 — Green Extension (Platoon Priority):
      If Camera 1 predicts a platoon of 4+ cars arriving in under 8s,
      dynamically extend Green by 5-10s to avoid stopping the wave.

    Rule 2 — Dynamic Balancing (Queue Dispersal):
      If Camera 2 senses East-West waiting queue is 2x larger than North-South,
      initiate safe amber transition.

    Rule 3 — Emergency Preemption:
      If an emergency vehicle / ambulance is logged at Camera 1, trigger
      immediate clearance of conflicting greens to ensure an open green corridor.
    """

    def __init__(self):
        self.phase = SignalPhase.NS_GREEN
        self.phase_start_time: Optional[float] = None
        self.current_phase_duration = config.DEFAULT_GREEN_TIME
        self.time_remaining = config.DEFAULT_GREEN_TIME
        self.extensions_granted = 0
        self.max_extensions = 2  # Max 2 extensions per cycle to prevent infinite green
        self.emergency_mode = False

        self.last_decision = "Normal Adaptive Cycle Running"
        self.decision_history: List[DecisionEvent] = []

        # Queue tracking simulation
        self.ns_queue_count = 2
        self.ew_queue_count = 3

    def log_decision(self, current_time: float, rule: str, action: str, reason: str, badge_type: str = "info"):
        event = DecisionEvent(
            timestamp=round(current_time, 1),
            rule=rule,
            phase=self.phase.value,
            action=action,
            reason=reason,
            badge_type=badge_type,
        )
        self.decision_history.insert(0, event)
        if len(self.decision_history) > 20:
            self.decision_history.pop()
        self.last_decision = f"[{rule}] {action}: {reason}"

    def update(
        self,
        current_time: float,
        active_platoons: list,
        active_approaching: list,
        camera2_queue_count: int = None,
        cross_queue_count: int = None,
    ):
        """
        Evaluates dynamic trajectory conditions and updates the traffic signal state machine.
        """
        if self.phase_start_time is None:
            self.phase_start_time = current_time

        elapsed = current_time - self.phase_start_time
        self.time_remaining = max(0.0, round(self.current_phase_duration - elapsed, 1))

        if camera2_queue_count is not None:
            self.ns_queue_count = max(0, camera2_queue_count)
        if cross_queue_count is not None:
            self.ew_queue_count = max(0, cross_queue_count)

        # -------------------------------------------------------------
        # RULE 3: Emergency Preemption
        # -------------------------------------------------------------
        has_emergency = any(v.get("is_emergency", False) for v in active_approaching)

        if has_emergency:
            self.emergency_mode = True
            if self.phase == SignalPhase.NS_GREEN:
                # Hold corridor green until ambulance clears
                self.current_phase_duration = max(self.current_phase_duration, elapsed + 12.0)
                self.time_remaining = round(self.current_phase_duration - elapsed, 1)
                self.log_decision(
                    current_time,
                    "Rule 3 — Emergency Preemption",
                    "CORRIDOR GREEN HOLD",
                    "Emergency vehicle detected at Upstream Camera 1. Holding open green corridor.",
                    "danger"
                )
                return
            elif self.phase in (SignalPhase.EW_GREEN, SignalPhase.EW_AMBER):
                # Rapid clearance of conflicting phases
                self.phase = SignalPhase.EW_AMBER
                self.phase_start_time = current_time
                self.current_phase_duration = 2.0  # Fast clearance
                self.time_remaining = 2.0
                self.log_decision(
                    current_time,
                    "Rule 3 — Emergency Preemption",
                    "RAPID CONFLICT CLEARANCE",
                    "Emergency vehicle approaching. Forcing rapid clearance of East-West phase.",
                    "danger"
                )
                return
        else:
            self.emergency_mode = False

        # -------------------------------------------------------------
        # RULE 1: Green Extension (Platoon Priority)
        # -------------------------------------------------------------
        if self.phase == SignalPhase.NS_GREEN and self.time_remaining <= 6.0:
            # Check if any incoming platoon has ETA < 8s and size >= 4
            eligible_platoon = None
            for p in active_platoons:
                if p.is_approaching and p.shared_eta <= config.PLATOON_MAX_ETA_TRIGGER and p.size >= config.PLATOON_MIN_SIZE:
                    eligible_platoon = p
                    break

            if eligible_platoon and self.extensions_granted < self.max_extensions:
                self.current_phase_duration += config.GREEN_EXTENSION_TIME
                self.time_remaining = round(self.current_phase_duration - elapsed, 1)
                self.extensions_granted += 1
                self.log_decision(
                    current_time,
                    "Rule 1 — Green Extension",
                    f"GREEN EXTENDED (+{int(config.GREEN_EXTENSION_TIME)}s)",
                    f"Platoon #{eligible_platoon.platoon_id} ({eligible_platoon.size} cars) approaching with ETA {eligible_platoon.shared_eta}s. Preserving continuous green wave.",
                    "success"
                )
                return

        # -------------------------------------------------------------
        # RULE 2: Dynamic Balancing (Queue Dispersal)
        # -------------------------------------------------------------
        if self.phase == SignalPhase.NS_GREEN and elapsed >= 10.0:
            # Check if East-West waiting queue is 2x larger than North-South corridor queue
            ratio = (self.ew_queue_count / max(1, self.ns_queue_count))
            if ratio >= config.QUEUE_BALANCE_RATIO and self.ew_queue_count >= config.QUEUE_CRITICAL_COUNT:
                # Trigger early amber transition for queue dispersal
                self.phase = SignalPhase.NS_AMBER
                self.phase_start_time = current_time
                self.current_phase_duration = config.DEFAULT_AMBER_TIME
                self.time_remaining = config.DEFAULT_AMBER_TIME
                self.extensions_granted = 0
                self.log_decision(
                    current_time,
                    "Rule 2 — Dynamic Balancing",
                    "EARLY AMBER TRANSITION",
                    f"Cross-traffic queue ratio {ratio:.1f}x (EW: {self.ew_queue_count} vs NS: {self.ns_queue_count}) exceeds threshold. Initiating queue dispersal.",
                    "warning"
                )
                return

        # -------------------------------------------------------------
        # Regular State Machine Transitions
        # -------------------------------------------------------------
        if self.time_remaining <= 0.0:
            self._transition_next_phase(current_time)

    def _transition_next_phase(self, current_time: float):
        if self.phase == SignalPhase.NS_GREEN:
            self.phase = SignalPhase.NS_AMBER
            self.current_phase_duration = config.DEFAULT_AMBER_TIME
            action = "Amber Clearance Transition"
        elif self.phase == SignalPhase.NS_AMBER:
            self.phase = SignalPhase.EW_GREEN
            self.current_phase_duration = config.DEFAULT_RED_TIME
            action = "Cross-Street Green Transition"
        elif self.phase == SignalPhase.EW_GREEN:
            self.phase = SignalPhase.EW_AMBER
            self.current_phase_duration = config.DEFAULT_AMBER_TIME
            action = "Cross-Street Amber Transition"
        else:  # EW_AMBER -> NS_GREEN
            self.phase = SignalPhase.NS_GREEN
            self.current_phase_duration = config.DEFAULT_GREEN_TIME
            self.extensions_granted = 0
            action = "Corridor Green Return"

        self.phase_start_time = current_time
        self.time_remaining = self.current_phase_duration
        self.log_decision(
            current_time,
            "State Machine",
            action,
            f"Phase progressed to {self.phase.value}.",
            "info"
        )

    def get_signal_display_info(self) -> dict:
        """
        Returns structured styling and state information for UI traffic light widget.
        """
        is_green = (self.phase == SignalPhase.NS_GREEN)
        is_amber = (self.phase in (SignalPhase.NS_AMBER, SignalPhase.EW_AMBER))
        is_red = (self.phase == SignalPhase.EW_GREEN)

        color_hex = "#22c55e" if is_green else ("#f59e0b" if is_amber else "#ef4444")
        phase_label = "CORRIDOR GREEN" if is_green else ("CLEARANCE AMBER" if is_amber else "CORRIDOR RED")

        return {
            "phase": self.phase.value,
            "phase_label": phase_label,
            "is_green": is_green,
            "is_amber": is_amber,
            "is_red": is_red,
            "color_hex": color_hex,
            "time_remaining": self.time_remaining,
            "extensions_granted": self.extensions_granted,
            "emergency_mode": self.emergency_mode,
            "ns_queue": self.ns_queue_count,
            "ew_queue": self.ew_queue_count,
            "last_decision": self.last_decision,
        }
