import torch
import torch.nn as nn
import torch.nn.functional as F


def project_to_simplex(v: torch.Tensor, z: float = 1.0) -> torch.Tensor:
    """Euclidean projection onto the probability simplex."""
    n_features = v.shape[-1]
    u, _ = torch.sort(v, descending=True, dim=-1)
    cssv = torch.cumsum(u, dim=-1)
    ind = torch.arange(1, n_features + 1, device=v.device, dtype=v.dtype)
    cond = u - (cssv - z) / ind > 0
    rho = torch.count_nonzero(cond, dim=-1, keepdim=True) - 1
    theta = (cssv.gather(-1, rho) - z) / (rho.to(v.dtype) + 1)
    w = torch.clamp(v - theta, min=0.0)
    return w


class MultiTheoryConstraintLattice(nn.Module):
    """
    Multi-Theory Constraint Lattice with Entropy-Regularized 
    Penalized Distribution Adjustment (Equations 5 & 6).
    """
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        num_strategies: int = 8,
        num_constraints: int = 6,
        rho: float = 1.0,
        kappa: float = 0.5,
        max_iter: int = 50,
        tol: float = 1e-4,
        lr: float = 0.05
    ):
        super().__init__()
        self.num_strategies = num_strategies
        self.num_constraints = num_constraints
        self.rho = rho
        self.kappa = kappa
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr

        self.gating = nn.Linear(hidden_dim + state_dim, 3)

        # Base operational constraint matrices for [PC, MI, CBT]
        # Dimensions: [num_theories (3), num_constraints (6), num_strategies (8)]
        C_init = torch.zeros(3, num_constraints, num_strategies)
        # Row 0: Exploration non-directivity (PC/MI: restrict advice at index 5)
        C_init[0, 0, 5] = 1.0
        C_init[1, 0, 5] = 0.8
        # Row 1: Low readiness autonomy protection (MI: restrict advice at index 5)
        C_init[1, 1, 5] = 1.0
        # Row 2: High affect reflection lower bound (PC: require reflection at index 2)
        C_init[0, 2, 2] = -1.0
        # Row 3: CBT collaborative reframing bound (CBT: regulate at index 4)
        C_init[2, 3, 4] = 0.5
        # Row 4 & 5: Invariant hard safety boundaries
        C_init[:, 4, :] = 0.0
        C_init[:, 5, :] = 0.0
        self.register_buffer("C_base", C_init)

        # Admissibility thresholds vector b
        b_init = torch.tensor([0.15, 0.20, -0.30, 0.25, 0.0, 0.0])
        self.register_buffer("b", b_init)

    def forward(
        self,
        h_t: torch.Tensor,
        z_t: torch.Tensor,
        p_0: torch.Tensor,
        U_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_t: Contextual hidden representation [B, hidden_dim]
            z_t: Continuous state vector [B, state_dim]
            p_0: Unconstrained strategy logits or prior distribution [B, num_strategies]
            U_t: Normalized stage uncertainty [B, 1]
        Returns:
            p_star: Feasible strategy distribution [B, num_strategies]
            alpha_t: Theory mixture weights [B, 3]
            C_bar: State-conditioned mixed constraint matrix [B, num_constraints, num_strategies]
        """
        batch_size = h_t.size(0)
        p_0 = F.softmax(p_0, dim=-1)

        # 1. State-conditioned dynamic theory weights
        gate_input = torch.cat([z_t, h_t], dim=-1)
        alpha_t = F.softmax(self.gating(gate_input), dim=-1)  # [B, 3]

        # 2. Convex combination of constraint matrices
        C_bar = torch.einsum("bk,kms->bms", alpha_t, self.C_base)  # [B, M, S]

        # 3. Soft slack allocation (bounded by kappa * U_t for soft rows)
        xi_t = torch.zeros(batch_size, self.num_constraints, device=h_t.device)
        xi_t[:, :4] = self.kappa * U_t.expand(-1, 4)

        # 4. Projected Gradient Descent on Simplex
        p = p_0.clone().detach().requires_grad_(True)
        b_expanded = self.b.unsqueeze(0).expand(batch_size, -1)

        for _ in range(self.max_iter):
            # Violation penalty: ||[C_bar * p - b - xi]_+||_2^2
            constraints_val = torch.bmm(C_bar, p.unsqueeze(-1)).squeeze(-1) - b_expanded - xi_t
            violations = F.relu(constraints_val)
            loss_penalty = self.rho * torch.sum(violations ** 2, dim=-1)

            # KL divergence: D_KL(p || p_0)
            loss_kl = F.kl_div(p.log().clamp(min=-100), p_0, reduction="batchmean")
            loss = loss_kl + loss_penalty.mean()

            grad = torch.autograd.grad(loss, p)[0]
            with torch.no_grad():
                p_next = project_to_simplex(p - self.lr * grad, z=1.0)
                diff = torch.norm(p_next - p, p=float("inf"), dim=-1).max()
                p.copy_(p_next)
                if diff < self.tol:
                    break

        return p.detach(), alpha_t, C_bar
