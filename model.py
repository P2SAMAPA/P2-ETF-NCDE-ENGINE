# model.py — Neural Controlled Differential Equation for ETF return forecasting
#
# Architecture:
#   1. VectorField f(h, t)  — MLP that maps hidden state h to dh/dt
#   2. NCDEModel            — runs the ODE solver over the asset+macro path
#   3. ReadoutHead          — MLP: h(T) → (mu, log_sigma) per ETF
#   4. NCDEForecaster       — wraps all three; forward() returns (mu, sigma)
#
# The "controlled" aspect: the asset+macro path X(t) enters via the vector field
# as f(h, X'(t)) where X'(t) = dX/dt from the cubic spline.
# Macro series act as an exogenous control signal shaping the hidden trajectory.
#
# Loss: Gaussian NLL — train to predict mu and sigma jointly.
# Confidence output: 1 / sigma (normalised in predict.py).

import torch
import torch.nn as nn
import torchcde
import config as cfg


# ── Vector field ───────────────────────────────────────────────────────────────

class VectorField(nn.Module):
    """
    f(h, X'(t)) — the right-hand side of dh = f(h, X(t)) dX(t).

    Input:  hidden state h  (hidden_dim,)
            path derivative (input_dim,)  — from torchcde spline
    Output: matrix of shape (hidden_dim, input_dim) — the "control matrix"

    torchcde calls this as f(t, h) and internally multiplies by dX/dt.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        layers = []
        in_dim = hidden_dim
        for i in range(n_layers):
            out_dim = cfg.VECTOR_FIELD_DIM if i < n_layers - 1 else hidden_dim * input_dim
            layers += [nn.Linear(in_dim, out_dim)]
            if i < n_layers - 1:
                layers += [nn.Tanh(), nn.Dropout(dropout)]
            in_dim = cfg.VECTOR_FIELD_DIM

        self.net = nn.Sequential(*layers)

    def forward(self, t, h):
        # h: (batch, hidden_dim)
        out = self.net(h)
        # Reshape to (batch, hidden_dim, input_dim) as required by torchcde
        return out.view(h.shape[0], self.hidden_dim, self.input_dim)


# ── NCDE model ─────────────────────────────────────────────────────────────────

class NCDEModel(nn.Module):
    """
    Runs the controlled ODE:  dh = f(h, X(t)) dX(t),  h(t0) = h0
    using torchdiffeq (via torchcde) with the adjoint method.

    Returns h(T) — the terminal hidden state for readout.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.vector_field = VectorField(input_dim, hidden_dim, n_layers, dropout)
        # Project first observation X(t0) → initial hidden state
        self.initial_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, X_path: torchcde.CubicSpline) -> torch.Tensor:
        """
        Args:
            X_path : torchcde.CubicSpline built from (batch, time, channels)
        Returns:
            h_T    : (batch, hidden_dim) terminal hidden state
        """
        # Initial hidden state from first path observation
        X0  = X_path.evaluate(X_path.interval[0])   # (batch, input_dim)
        h0  = self.initial_proj(X0)                  # (batch, hidden_dim)

        # Solve the CDE
        # For fixed-step solvers (euler, midpoint): pass explicit time grid via t.
        # For adaptive solvers (dopri5): rtol/atol control accuracy.
        T      = cfg.LOOKBACK
        t_span = torch.linspace(0, T - 1, cfg.ODE_STEPS + 1)

        solver_kwargs = {}
        if cfg.SOLVER in ("euler", "midpoint", "rk4"):
            solve_t = t_span
        else:
            solve_t      = X_path.interval
            solver_kwargs = {"rtol": 1e-3, "atol": 1e-5}

        h_T = torchcde.cdeint(
            X=X_path,
            func=self.vector_field,
            z0=h0,
            t=solve_t,
            adjoint=cfg.ADJOINT,
            method=cfg.SOLVER,
            **solver_kwargs,
        )
        # h_T shape: (batch, 2, hidden_dim) — take terminal state
        return h_T[:, -1, :]


# ── Readout head ───────────────────────────────────────────────────────────────

class ReadoutHead(nn.Module):
    """
    MLP: h(T) → (mu, log_sigma) per ETF.

    mu       = predicted next-day return
    sigma    = predicted uncertainty (exp(log_sigma) for positivity)
    confidence (in predict.py) = 1 / sigma, normalised across ETFs
    """

    def __init__(self, hidden_dim: int, readout_dim: int, n_assets: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, readout_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_dim, readout_dim // 2),
            nn.SiLU(),
            nn.Linear(readout_dim // 2, n_assets * 2),   # mu + log_sigma per asset
        )
        self.n_assets = n_assets

    def forward(self, h: torch.Tensor):
        # h: (batch, hidden_dim)
        out       = self.net(h)                          # (batch, n_assets * 2)
        mu        = out[:, :self.n_assets]               # (batch, n_assets)
        log_sigma = out[:, self.n_assets:]               # (batch, n_assets)
        sigma     = torch.exp(log_sigma.clamp(-6, 2))    # clamp for stability
        return mu, sigma


# ── Full forecaster ────────────────────────────────────────────────────────────

class NCDEForecaster(nn.Module):
    """
    End-to-end NCDE forecaster.

    forward() accepts pre-built torchcde spline paths and returns (mu, sigma).
    The caller (train.py / predict.py) is responsible for building the splines
    via features.build_ncde_path().
    """

    def __init__(
        self,
        n_asset_path_dim: int,
        n_macro_feats:    int,
        n_assets:         int,
        hidden_dim:       int  = None,
        vector_field_dim: int  = None,
        n_layers:         int  = None,
        readout_dim:      int  = None,
        dropout:          float = None,
    ):
        super().__init__()
        hidden_dim       = hidden_dim       or cfg.HIDDEN_DIM
        n_layers         = n_layers         or cfg.N_LAYERS
        readout_dim      = readout_dim      or cfg.READOUT_DIM
        dropout          = dropout          or cfg.DROPOUT

        # Combined input dim: asset path + macro control concatenated channel-wise
        input_dim = n_asset_path_dim + n_macro_feats

        self.ncde    = NCDEModel(input_dim, hidden_dim, n_layers, dropout)
        self.readout = ReadoutHead(hidden_dim, readout_dim, n_assets, dropout)

        # Store for serialisation
        self.n_asset_path_dim = n_asset_path_dim
        self.n_macro_feats    = n_macro_feats
        self.n_assets         = n_assets
        self.hidden_dim       = hidden_dim

    def forward(self, X_path: torchcde.CubicSpline):
        """
        Args:
            X_path : CubicSpline built from asset + macro concatenated channel-wise,
                     shape (batch, time, n_asset_path_dim + n_macro_feats).
                     Build with build_combined_path() in train.py / predict.py.

        Returns:
            mu    : (batch, n_assets) — predicted next-day returns
            sigma : (batch, n_assets) — predicted uncertainty
        """
        h_T       = self.ncde(X_path)
        mu, sigma = self.readout(h_T)
        return mu, sigma


# ── Loss function ──────────────────────────────────────────────────────────────

def gaussian_nll_loss(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Gaussian negative log-likelihood loss.

    Trains the model to jointly predict the mean and uncertainty of next-day returns.
    Lower sigma = higher confidence; the model is penalised for being confidently wrong.

    Args:
        mu    : (batch, n_assets) predicted means
        sigma : (batch, n_assets) predicted std deviations (must be > 0)
        y     : (batch, n_assets) actual next-day returns

    Returns:
        scalar mean NLL loss
    """
    dist = torch.distributions.Normal(mu, sigma)
    return -dist.log_prob(y).mean()
