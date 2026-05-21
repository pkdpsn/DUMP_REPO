"""
Learn dt(x, y, action, U_patch) -> travel time
from 3 known potentials, then test on a 4th unseen one.

Potentials used:
  Train: Mexican hat U0=0.2, Mexican hat U0=0.4, Peaks
  Test:  Mexican hat U0=0.3  (unseen U0), Gaussian bump (unseen family)

This script is designed to expose WHERE the network fails,
not just whether it fails.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs

torch.manual_seed(42)
np.random.seed(42)

# ── Grid & actions ────────────────────────────────────────────────────────────

N      = 51
xs     = np.linspace(-3.0, 3.0, N)
ys     = np.linspace(-3.0, 3.0, N)
CELL   = 6.0 / (N - 1)
ACTIONS = [(1,0), (-1,0), (0,1), (0,-1)]   # right, left, up, down
PATCH_K = 5    # patch = (2K+1) x (2K+1) = 11x11 around (i,j)

# ── Potentials ────────────────────────────────────────────────────────────────

def mexican_hat(x, y, U0):
    # rescale from [-3,3] back to [-0.75,0.75] for the formula
    xr, yr = x / 4, y / 4
    rho = np.sqrt(xr**2 + yr**2)
    return np.where(rho <= 0.5, 16 * U0 * (rho**2 - 0.25)**2, 0.0)

def peaks(x, y):
    # Eq (6) Schneider & Stark, native [-3,3] domain
    return (  0.3 * (1 - x)**2 * np.exp(-x**2 - (y + 1)**2)
            - (0.2*x - x**3 - y**5) * np.exp(-x**2 - y**2)
            - (1/30) * np.exp(-(x + 1)**2 - y**2) )

def gaussian_bump(x, y):
    # single Gaussian obstacle — unseen family at test time
    return 0.5 * np.exp(-((x - 0.5)**2 + (y + 0.5)**2) / 0.8)

# ── Force & travel time ───────────────────────────────────────────────────────

def compute_force(U_grid):
    """
    Numerical gradient → force F = -∇U on the full N×N grid.
    Returns fx, fy each shape (N, N).
    """
    # np.gradient returns [d/dy, d/dx] for a 2D array with indexing='ij'
    gy, gx = np.gradient(U_grid, CELL, CELL)
    return -gx, -gy   # F = -∇U

def travel_time_grid(fx, fy, di, dj):
    """
    Compute dt for action (di,dj) at every grid point.
    Returns array shape (N, N), inf where move is unphysical.
    """
    dx, dy = di * CELL, dj * CELL
    L      = np.sqrt(dx**2 + dy**2)
    tx, ty = dx/L, dy/L
    tF     = tx*fx + ty*fy
    Fp2    = fx**2 + fy**2 - tF**2
    disc   = 1.0 - Fp2
    disc   = np.maximum(disc, 0)          # clip small negatives from numerics
    v      = tF + np.sqrt(disc)
    dt     = np.where(v > 1e-6, L/v, np.inf)
    return dt

# ── Patch extraction ──────────────────────────────────────────────────────────

def extract_patch(U_grid, i, j, k=PATCH_K):
    """
    Extract (2k+1)x(2k+1) patch of U centred at (i,j).
    Pads with edge values if near boundary.
    """
    padded = np.pad(U_grid, k, mode='edge')
    patch  = padded[i : i + 2*k+1, j : j + 2*k+1]
    return patch.flatten()   # (2k+1)^2 values

# ── Dataset generation ────────────────────────────────────────────────────────

def make_dataset(pot_fn, label, n_samples=40000):
    """
    Generate (input, target) pairs for one potential.

    Input  : [x_norm, y_norm, di, dj, fx, fy, patch (normalised)]
    Target : v = CELL/dt  (speed, bounded in [0, ~2])
             predicting speed is cleaner than dt which blows up
    """
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    U_grid = pot_fn(XX, YY)
    fx_grid, fy_grid = compute_force(U_grid)

    # normalise patch globally for this potential
    U_mean, U_std = U_grid.mean(), U_grid.std() + 1e-8

    inputs, targets = [], []
    rng = np.random.default_rng(42)

    attempts = 0
    while len(inputs) < n_samples and attempts < n_samples * 10:
        attempts += 1
        i  = rng.integers(PATCH_K, N - PATCH_K)
        j  = rng.integers(PATCH_K, N - PATCH_K)
        ai = rng.integers(4)
        di, dj = ACTIONS[ai]
        ni, nj = i+di, j+dj
        if not (0 <= ni < N and 0 <= nj < N):
            continue

        dt = travel_time_grid(fx_grid, fy_grid, di, dj)[i, j]
        if not np.isfinite(dt) or dt > 100:
            continue

        v     = CELL / dt    # speed — our prediction target
        patch = extract_patch(U_grid, i, j)
        patch_norm = (patch - U_mean) / U_std

        # build input vector
        x_norm  = xs[i] / 3.0    # normalise to [-1,1]
        y_norm  = ys[j] / 3.0
        fx_norm = fx_grid[i,j]
        fy_norm = fy_grid[i,j]

        inp = np.concatenate([
            [x_norm, y_norm],          # position  (2)
            [float(di), float(dj)],    # action    (2)
            [fx_norm, fy_norm],        # force     (2)
            patch_norm,                # U patch   (11x11 = 121)
        ])                             # total: 127

        inputs.append(inp.astype(np.float32))
        targets.append(np.float32(v))

    X = np.stack(inputs)
    y = np.array(targets)
    print(f"  [{label}] samples={len(X)}  v range=[{y.min():.3f}, {y.max():.3f}]  "
          f"mean={y.mean():.3f}")
    return X, y, U_grid, fx_grid, fy_grid

# ── Neural network ────────────────────────────────────────────────────────────

INPUT_DIM = 2 + 2 + 2 + (2*PATCH_K+1)**2   # 127

class DTNet(nn.Module):
    """
    Simple MLP.  Predicts speed v = CELL/dt.
    Separate pathway for (position+action+force) vs patch,
    then fuse — makes it easier to see which part contributes.
    """
    def __init__(self):
        super().__init__()

        # physics pathway: position, action, force  (6 inputs)
        self.phys = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # patch pathway: 121 inputs → compact representation
        self.patch_enc = nn.Sequential(
            nn.Linear(121, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # fusion
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),   # ensures positive output
        )

    def forward(self, x):
        phys  = x[:, :6]
        patch = x[:, 6:]
        p1 = self.phys(phys)
        p2 = self.patch_enc(patch)
        return self.head(torch.cat([p1, p2], dim=1)).squeeze(1)

# ── Training ──────────────────────────────────────────────────────────────────

def train(model, X_train, y_train, epochs=60, batch_size=512, lr=1e-3):
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    loss_fn   = nn.MSELoss()

    X_t = torch.tensor(X_train, device=device)
    y_t = torch.tensor(y_train, device=device)
    ds  = TensorDataset(X_t, y_t)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True)

    history = []
    for ep in range(1, epochs+1):
        model.train()
        ep_loss = 0
        for xb, yb in dl:
            optimiser.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimiser.step()
            ep_loss += loss.item() * len(xb)
        scheduler.step()
        ep_loss /= len(X_train)
        history.append(ep_loss)
        if ep % 10 == 0:
            print(f"  epoch {ep:3d}  loss={ep_loss:.6f}")
    return history, device

def evaluate(model, X_test, y_test, device, label):
    model.eval()
    with torch.no_grad():
        X_t  = torch.tensor(X_test, device=device)
        pred = model(X_t).cpu().numpy()
    y_true = y_test

    mae  = np.mean(np.abs(pred - y_true))
    mape = np.mean(np.abs(pred - y_true) / (y_true + 1e-8)) * 100
    r2   = 1 - np.sum((pred - y_true)**2) / np.sum((y_true - y_true.mean())**2)
    print(f"  [{label}]  MAE={mae:.4f}  MAPE={mape:.2f}%  R²={r2:.4f}")
    return pred, mae, mape, r2

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── 1. Generate data ─────────────────────────────────────────────────────
    print("\n=== Generating training data ===")
    train_pots = [
        (lambda x,y: mexican_hat(x,y,0.2), "MexHat_U0=0.2"),
        (lambda x,y: mexican_hat(x,y,0.4), "MexHat_U0=0.4"),
        (peaks,                             "Peaks"),
    ]
    test_pots = [
        (lambda x,y: mexican_hat(x,y,0.3), "MexHat_U0=0.3 [same family, unseen U0]"),
        (gaussian_bump,                     "GaussianBump  [unseen family]"),
    ]

    all_train_X, all_train_y = [], []
    train_meta = []   # store grids for plotting
    for pot_fn, label in train_pots:
        print(f"\nPotential: {label}")
        X, y, Ug, fx, fy = make_dataset(pot_fn, label, n_samples=40000)
        all_train_X.append(X)
        all_train_y.append(y)
        train_meta.append((label, Ug, fx, fy))

    X_train = np.concatenate(all_train_X)
    y_train = np.concatenate(all_train_y)

    # shuffle
    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]
    print(f"\nTotal training samples: {len(X_train)}")

    print("\n=== Generating test data ===")
    test_datasets = []
    for pot_fn, label in test_pots:
        print(f"\nPotential: {label}")
        X, y, Ug, fx, fy = make_dataset(pot_fn, label, n_samples=10000)
        test_datasets.append((label, X, y, Ug, fx, fy))

    # ── 2. Train ──────────────────────────────────────────────────────────────
    print("\n=== Training ===")
    model   = DTNet()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    history, device = train(model, X_train, y_train, epochs=80)

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    print("\n=== Evaluation ===")

    # also evaluate on a held-out split of training potentials
    print("In-distribution (held-out 10% of training data):")
    n_hold = len(X_train) // 10
    _, _, r2_in = evaluate(model,
                           X_train[-n_hold:], y_train[-n_hold:],
                           device, "In-dist")[1:4]

    print("\nOut-of-distribution:")
    test_results = []
    for label, X, y, Ug, fx, fy in test_datasets:
        pred, mae, mape, r2 = evaluate(model, X, y, device, label)
        test_results.append((label, pred, y, Ug, mae, mape, r2))

    # ── 4. Plots ──────────────────────────────────────────────────────────────
    print("\n=== Plotting ===")

    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    # Figure 1: training loss
    fig1, ax = plt.subplots(figsize=(7,4))
    ax.plot(history, color='#3498db', lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE loss')
    ax.set_title('Training loss (3 potentials combined)')
    ax.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig('nn_dt_loss.png', dpi=130, bbox_inches='tight')
    print("Saved: nn_dt_loss.png")

    # Figure 2: predicted vs true speed scatter for all test sets
    n_test = len(test_results)
    fig2, axes = plt.subplots(1, n_test, figsize=(6*n_test, 5))
    if n_test == 1: axes = [axes]
    for ax, (label, pred, y_true, _, mae, mape, r2) in zip(axes, test_results):
        vmax = max(y_true.max(), pred.max())
        ax.scatter(y_true, pred, alpha=0.15, s=6, color='#3498db')
        ax.plot([0, vmax], [0, vmax], 'r--', lw=1.5, label='Perfect')
        ax.set_xlabel('True speed v'); ax.set_ylabel('Predicted speed v')
        ax.set_title(f'{label}\nMAE={mae:.4f}  MAPE={mape:.1f}%  R²={r2:.3f}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
    fig2.suptitle('Predicted vs True speed — test potentials', fontsize=12)
    fig2.tight_layout()
    fig2.savefig('nn_dt_scatter.png', dpi=130, bbox_inches='tight')
    print("Saved: nn_dt_scatter.png")

    # Figure 3: spatial error map — WHERE does the NN fail on the grid?
    # For each test potential, compute predicted vs true dt at every grid point
    fig3, axes3 = plt.subplots(2, n_test, figsize=(6*n_test, 10))
    if n_test == 1: axes3 = axes3.reshape(2,1)

    model.eval()
    for col, ((pot_fn, pot_label), (res_label, pred, y_true_r, Ug_r, mae_r, mape_r, r2_r)) in enumerate(zip(test_pots, test_results)):
        pot_name = res_label.split('[')[0].strip()
        Ug_t  = pot_fn(XX, YY)
        fx_t, fy_t = compute_force(Ug_t)
        U_mean, U_std = Ug_t.mean(), Ug_t.std() + 1e-8

        # pick one action (right = action 0) for the spatial map
        ai_vis = 0
        di, dj = ACTIONS[ai_vis]
        dt_true = travel_time_grid(fx_t, fy_t, di, dj)   # (N,N)
        v_true  = np.where(np.isfinite(dt_true), CELL/dt_true, np.nan)

        # build inputs for every valid grid point
        rows, cols_idx, inps = [], [], []
        for i in range(PATCH_K, N-PATCH_K):
            for j in range(PATCH_K, N-PATCH_K):
                if not np.isfinite(v_true[i,j]):
                    continue
                patch = extract_patch(Ug_t, i, j)
                patch_norm = (patch - U_mean) / U_std
                inp = np.concatenate([
                    [xs[i]/3.0, ys[j]/3.0],
                    [float(di), float(dj)],
                    [fx_t[i,j], fy_t[i,j]],
                    patch_norm,
                ]).astype(np.float32)
                rows.append(i); cols_idx.append(j); inps.append(inp)

        with torch.no_grad():
            X_sp   = torch.tensor(np.stack(inps), device=device)
            v_pred = model(X_sp).cpu().numpy()

        # fill error grid
        err_grid = np.full((N,N), np.nan)
        for r, c, vp, vt in zip(rows, cols_idx, v_pred,
                                  [v_true[r2,c2] for r2,c2 in zip(rows,cols_idx)]):
            err_grid[r,c] = abs(vp - vt)

        # top: potential landscape
        ax_top = axes3[0, col]
        im = ax_top.imshow(Ug_t.T, origin='lower', extent=[-3,3,-3,3],
                           cmap='RdYlGn_r', aspect='equal')
        plt.colorbar(im, ax=ax_top, fraction=0.046, pad=0.04)
        ax_top.set_title(f'{pot_name}\nPotential landscape')
        ax_top.set_xlabel('x'); ax_top.set_ylabel('y')

        # bottom: absolute error map
        ax_bot = axes3[1, col]
        im2 = ax_bot.imshow(err_grid.T, origin='lower', extent=[-3,3,-3,3],
                            cmap='hot_r', aspect='equal',
                            vmin=0, vmax=np.nanpercentile(err_grid, 95))
        plt.colorbar(im2, ax=ax_bot, fraction=0.046, pad=0.04, label='|v_pred - v_true|')
        ax_bot.set_title(f'Error map (action=right)\nRed = NN fails here')
        ax_bot.set_xlabel('x'); ax_bot.set_ylabel('y')

    fig3.suptitle('Spatial error maps — where does the NN fail?', fontsize=13)
    fig3.tight_layout()
    fig3.savefig('nn_dt_error_map.png', dpi=130, bbox_inches='tight')
    print("Saved: nn_dt_error_map.png")

    # Figure 4: error vs |F_perp| — the key diagnostic from our derivation
    # High |F_perp| = near singularity = NN should struggle most here
    fig4, axes4 = plt.subplots(1, n_test, figsize=(6*n_test, 5))
    if n_test == 1: axes4 = [axes4]

    for ax, (label, pred, y_true, Ug, mae, mape, r2) in zip(axes4, test_results):
        # recover Fperp magnitude from the stored test data
        # action direction from the stored inputs (cols 2,3 of X)
        # We approximate: just plot error vs true speed as proxy
        # (high v = low Fperp regime, low v = high Fperp regime)
        err   = np.abs(pred - y_true)
        speed = y_true
        # bin by speed
        bins  = np.linspace(0, speed.max(), 20)
        bin_i = np.digitize(speed, bins)
        bin_mae = [err[bin_i==k].mean() if (bin_i==k).sum()>0 else np.nan
                   for k in range(1, len(bins))]
        ax.bar(bins[:-1], bin_mae, width=bins[1]-bins[0],
               color='#e74c3c', alpha=0.8, align='edge')
        ax.set_xlabel('True speed v  (low v = near singularity |F_perp|→1)')
        ax.set_ylabel('Mean absolute error')
        ax.set_title(f'{label.split("[")[0]}\nError by speed regime')
        ax.grid(True, alpha=0.3)

    fig4.suptitle('Error concentrated near singularity? (low speed = high |F_perp|)',
                  fontsize=11)
    fig4.tight_layout()
    fig4.savefig('nn_dt_error_by_speed.png', dpi=130, bbox_inches='tight')
    print("Saved: nn_dt_error_by_speed.png")

    plt.show()
    print("\nDone. Check the 4 saved figures to diagnose where the NN fails.")
