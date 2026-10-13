from dataclasses import dataclass
import torch


@dataclass
class RiskGateOutput:
    is_escalated: bool
    response_text: str | None
    bypass_generation: bool


class BinaryRiskGate:
    """Pre-generation short-circuit early-exit risk gate."""
    def __init__(self, tau_risk: float = 0.65, fallback_template: str | None = None):
        self.tau_risk = tau_risk
        self.fallback_template = fallback_template or (
            "It sounds like you are going through a critical moment. "
            "Please reach out to a professional counselor or local emergency services."
        )

    def evaluate_step(self, q_t: torch.Tensor) -> list[RiskGateOutput]:
        """
        Evaluates risk coordinates before autoregressive decoding.
        Args:
            q_t: Risk coordinate tensor [B, 1]
        """
        results = []
        for risk in q_t.squeeze(-1).tolist():
            if risk >= self.tau_risk:
                results.append(RiskGateOutput(
                    is_escalated=True,
                    response_text=self.fallback_template,
                    bypass_generation=True
                ))
            else:
                results.append(RiskGateOutput(
                    is_escalated=False,
                    response_text=None,
                    bypass_generation=False
                ))
        return results
