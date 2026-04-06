# model.py — Neural Controlled Differential Equation for ETF return forecasting
#
# Changes vs previous version:
#   - NCDEForecaster now accepts vector_field_dim as a constructor argument
#     so per-option configs in config.py are fully respected.
#   - VectorField layer dimension bug fixed (intermediate layers used wrong in_dim).
#   - LayerNorm inside VectorField to stabilise path dynamics across feature scales.
#   - ReadoutHead bottleneck removed; depth kept at 3 layers without squeeze.
#   - initial_proj uses Tanh to keep h0 bounded before ODE integration.

import torch
import torch.nn as nn
import torchcde

import config as cfg


# ── Vector field ───────────────────────────────────────────────────────────────

class VectorField(nn.Module):
    """
    f(h, X'(t)) — right-hand side of dh = f(h, X(t)) dX(t).

    Output shape: (batch, hidden_dim, input_dim) as required by torchcde.
    LayerNorm after each Tanh stabilises training when input channels span
    very different scales (e.g. VIX ~15–80 vs log-returns ~0.001).
    """

    def __init__(
        self,
        input_dim:       int,
        hidden_dim:      int,
        vector_field_dim: int,
        n_layers:        int,
        dropout:         float,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        out_dim = hidden_dim * input_dim
        layers  = []
        in_dim  = hidden_dim

        for i in range(n_layers):
            is_last   = (i == n_layers - 1)
            layer_out = out_dim if is_last else vector_field_dim
            layers.append(nn.Linear(in_dim, layer_out))
            if not is_last:
                layers.append(nn.Tanh())
                layers.append(nn.LayerNorm(vector_field_dim))
                layers.append(nn.Dropout(dropout))
            in_dim = vector_field_dim

        self.net = nn.Sequential(*layers)

    def forward(self, t, h):
        out = self.net(h)
        return out.view(h.shape[0], self.hidden_dim, self.input_dim)


# ── NCDE model ─────────────────────────────────────────────────────────────────

class NCDEModel(nn.Module):
    """
    Runs dh = f(h, X(t)) dX(t), h(t0) = h0.
    Returns h(T) — terminal hidden state.
    """

    def __init__(
        self,
        input_dim:        int,
        hidden_dim:       int,
        vector_field_dim: int,
        n_layers:         int,
        dropout:          float,
        solver:           str,
        adjoint:          bool,
        ode_steps:        int,
        lookback:         int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.solver     = solver
        self.adjoint    = adjoint
        self.ode_steps  = ode_steps
        self.lookback   = lookback

        self.vector_field = VectorField(
            input_dim, hidden_dim, vector_field_dim, n_layers, dropout
        )
        self.initial_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, X_path: torchcde.CubicSpline) -> torch.Tensor:
        X0 = X_path.evaluate(X_path.interval[0])
        h0 = self.initial_proj(X0)

        t_span = torch.linspace(0, self.lookback - 1, self.ode_steps + 1)

        if self.solver in ("euler", "midpoint", "rk4"):
            solve_t       = t_span
            solver_kwargs = {}
        else:
            solve_t       = X_path.interval
            solver_kwargs = {"rtol": 1e-3, "atol": 1e-5}

        h_T = torchcde.cdeint(
            X       = X_path,
            func    = self.vector_field,
            z0      = h0,
            t       = solve_t,
            adjoint = self.adjoint,
            method  = self.solver,
            **solver_kwargs,
        )
        return h_T[:, -1, :]


# ── Readout head ───────────────────────────────────────────────────────────────

class ReadoutHead(nn.Module):
    """
    h(T) → (mu, log_sigma) per ETF.

    Three layers without a bottleneck squeeze.
    LayerNorm on h(T) normalises the terminal state distribution before readout.
    """

    def __init__(self, hidden_dim: int, readout_dim: int, n_assets: int, dropout: float):
        super().__init__()
        self.norm     = nn.LayerNorm(hidden_dim)
        self.net      = nn.Sequential(
            nn.Linear(hidden_dim,  readout_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_dim, readout_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(readout_dim, n_assets * 2),
        )
        self.n_assets = n_assets

    def forward(self, h: torch.Tensor):
        h         = self.norm(h)
        out       = self.net(h)
        mu        = out[:, :self.n_assets]
        log_sigma = out[:, self.n_assets:]
        sigma     = torch.exp(log_sigma.clamp(-6, 2))
        return mu, sigma


# ── Full forecaster ────────────────────────────────────────────────────────────

class NCDEForecaster(nn.Module):
    """
    End-to-end NCDE forecaster. All architecture hyperparameters are passed
    explicitly so per-option configs from config.py are fully respected.
    """

    def __init__(
        self,
        n_asset_path_dim: int,
        n_macro_feats:    int,
        n_assets:         int,
        hidden_dim:       int   = None,
        vector_field_dim: int   = None,
        n_layers:         int   = None,
        readout_dim:      int   = None,
        dropout:          float = None,
        solver:           str   = None,
        adjoint:          bool  = None,
        ode_steps:        int   = None,
        lookback:         int   = None,
    ):
        super().__init__()

        # Fall back to flat cfg defaults (Option B values) if not provided
        hidden_dim       = hidden_dim       or cfg.HIDDEN_DIM
        vector_field_dim = vector_field_dim or cfg.VECTOR_FIELD_DIM
        n_layers         = n_layers         or cfg.N_LAYERS
        readout_dim      = readout_dim      or cfg.READOUT_DIM
        dropout          = dropout          or cfg.DROPOUT
        solver           = solver           or cfg.SOLVER
        adjoint          = adjoint          if adjoint is not None else cfg.ADJOINT
        ode_steps        = ode_steps        or cfg.ODE_STEPS
        lookback         = lookback         or cfg.LOOKBACK

        input_dim = n_asset_path_dim + n_macro_feats

        self.ncde = NCDEModel(
            input_dim, hidden_dim, vector_field_dim,
            n_layers, dropout, solver, adjoint, ode_steps, lookback,
        )
        self.readout = ReadoutHead(hidden_dim, readout_dim, n_assets, dropout)

        self.n_asset_path_dim = n_asset_path_dim
        self.n_macro_feats    = n_macro_feats
        self.n_assets         = n_assets
        self.hidden_dim       = hidden_dim

    def forward(self, X_path: torchcde.CubicSpline):
        h_T       = self.ncde(X_path)
        mu, sigma = self.readout(h_T)
        return mu, sigma


# ── Loss ───────────────────────────────────────────────────────────────────────

def gaussian_nll_loss(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Gaussian NLL. Model is penalised for being confidently wrong.
    Lower sigma = higher confidence in predict.py.
    """
    dist = torch.distributions.Normal(mu, sigma)
    return -dist.log_prob(y).mean()
