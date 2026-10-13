import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DurationAwareStateEstimator(nn.Module):
    """
    Infers latent state z_t = [e_t, c_t, n_t, r_t, q_t],
    discrete stage g_t, duration d_t, and normalized uncertainty U_t.
    """
    def __init__(self, hidden_dim: int, num_stages: int = 3, num_emotions: int = 32, max_duration: int = 15):
        super().__init__()
        self.num_stages = num_stages
        self.max_duration = max_duration

        # Latent continuous state heads
        self.head_affect = nn.Linear(hidden_dim, num_emotions)
        self.head_cause = nn.Linear(hidden_dim, 64)
        self.head_need = nn.Linear(hidden_dim, 64)
        self.head_readiness = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.head_risk = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

        # Semi-Markov stage and duration heads
        self.head_stage = nn.Linear(hidden_dim + 64*2 + num_emotions + 2, num_stages)
        self.head_duration = nn.Linear(hidden_dim + num_stages, max_duration)

    def forward(self, h_t: torch.Tensor) -> dict[str, torch.Tensor]:
        e_logits = self.head_affect(h_t)
        e_t = F.softmax(e_logits, dim=-1)
        c_t = self.head_cause(h_t)
        n_t = self.head_need(h_t)
        r_t = self.head_readiness(h_t)
        q_t = self.head_risk(h_t)

        z_t = torch.cat([e_t, c_t, n_t, r_t, q_t], dim=-1)

        stage_logits = self.head_stage(torch.cat([h_t, z_t], dim=-1))
        stage_probs = F.softmax(stage_logits, dim=-1)

        dur_input = torch.cat([h_t, stage_probs], dim=-1)
        dur_logits = self.head_duration(dur_input)
        dur_probs = F.softmax(dur_logits, dim=-1)

        # Normalized Shannon entropy over discrete stages: H(g_t) / log(3)
        entropy = -torch.sum(stage_probs * torch.log(stage_probs + 1e-9), dim=-1, keepdim=True)
        U_t = entropy / math.log(self.num_stages)

        return {
            "z_t": z_t,
            "r_t": r_t,
            "q_t": q_t,
            "stage_probs": stage_probs,
            "duration_probs": dur_probs,
            "U_t": U_t,
            "e_logits": e_logits,
            "c_t": c_t,
            "n_t": n_t
        }
