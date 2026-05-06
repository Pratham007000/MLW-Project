

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import h5py
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.utils.data import TensorDataset, DataLoader

# ── CHOOSE MODE ────────────────────────────────────────────────────────────────
PLOT_MODE = 'hybrid'   # 'paper' | 'hybrid'

# ── CONFIG ────────────────────────────────────────────────────────────────────
QUADRIGA_PATH = '/kaggle/input/datasets/anchitwork/quadriga/test_adp.mat'
MODEL_PATH    = '/kaggle/input/datasets/prathamarunshetty/csi-l2o-final-weights/best_csi_l2o_cr16.pth'

Na, Nt    = 32, 32
Ni        = 256
TOP_G     = 51
T_ITERS   = 10
BATCH     = 200
H_DIM     = 2 * Na * Nt
TRAINED_M = H_DIM // 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CR_LABELS  = ['1/64', '1/32', '1/16', '1/8', '1/4']
CR_VALUES  = [1/64,   1/32,   1/16,   1/8,   1/4 ]
CR_M       = [H_DIM//64, H_DIM//32, H_DIM//16, H_DIM//8, H_DIM//4]

PAPER_NMSE_LINEAR = {
    'ISTA':     [1.00,  0.87,  0.75,  0.52,  0.37],
    'MS4L2O':   [0.50,  0.42,  0.35,  0.44,  0.32],
    'CsiNet':   [0.50,  0.37,  0.27,  0.21,  0.185],
    'TiLISTA':  [0.50,  0.32,  0.22,  0.175, 0.14],
    'TransNet': [0.35,  0.25,  0.12,  0.083, 0.065],
    'Proposed': [0.44,  0.32,  0.185, 0.16,  0.085],
}

class AngularSparseTransform(nn.Module):
    def __init__(self, row_dim=2*Nt, Ni=256, top_G=51):
        super().__init__()
        self.top_G = top_G; self.Ni = Ni
        self.f_t = nn.Sequential(
            nn.Linear(row_dim, 128), nn.LeakyReLU(0.1),
            nn.Linear(128, 128),     nn.LeakyReLU(0.1),
            nn.Linear(128, Ni))
        self.f_inv_net = nn.Sequential(
            nn.Linear(Ni, 128),  nn.LeakyReLU(0.1),
            nn.Linear(128, 128), nn.LeakyReLU(0.1),
            nn.Linear(128, row_dim))

    def forward(self, h_mat):
        B, Na_dim, row_dim = h_mat.shape
        flat = h_mat.reshape(B * Na_dim, row_dim)
        z    = self.f_t(flat)
        if self.top_G < self.Ni:
            thr = z.abs().topk(self.top_G, dim=1).values.min(dim=1, keepdim=True).values
            z   = z * (z.abs() >= thr).float()
        return self.f_inv_net(z).reshape(B, Na_dim, row_dim)


class L2OModule(nn.Module):
    def __init__(self, signal_dim, T_iterations=10):
        super().__init__()
        self.D = signal_dim; self.T = T_iterations
        self.lstm      = nn.LSTM(input_size=2, hidden_size=2, num_layers=2, batch_first=True)
        self.inter_mlp = nn.Sequential(nn.Linear(2, 20), nn.Tanh(), nn.Linear(20, 20))
        self.mlp_p = nn.Linear(20, 1); self.mlp_a = nn.Linear(20, 1)
        self.mlp_b = nn.Linear(20, 1); self.mlp_theta = nn.Linear(20, 1)
        nn.init.constant_(self.mlp_theta.bias, -4.0)

    def _soft_threshold(self, u, theta):
        return u.sign() * F.relu(u.abs() - theta)

    def forward(self, s, W):
        B, D = s.shape[0], self.D; dev = s.device
        x = s @ W; y = x.clone(); BD = B * D
        h_st = torch.zeros(2, BD, 2, device=dev)
        c_st = torch.zeros(2, BD, 2, device=dev)
        for _ in range(self.T):
            grad_x = (x @ W.t() - s) @ W
            grad_y = (y @ W.t() - s) @ W
            inp    = torch.stack([x.reshape(BD), grad_x.reshape(BD)], dim=-1).unsqueeze(1)
            out, (h_st, c_st) = self.lstm(inp, (h_st, c_st))
            feat  = self.inter_mlp(out.squeeze(1))
            p     = torch.sigmoid(self.mlp_p(feat)).reshape(B, D) * 0.1
            a     = torch.sigmoid(self.mlp_a(feat)).reshape(B, D)
            b     = torch.sigmoid(self.mlp_b(feat)).reshape(B, D)
            theta = torch.exp(self.mlp_theta(feat)).reshape(B, D).clamp(1e-6, 1.0)
            x_hat = x - p * grad_x; y_hat = y - p * grad_y
            x_new = self._soft_threshold((1 - b) * x_hat + b * y_hat, theta)
            y_new = x_new + a * (x_new - x)
            x, y  = x_new, y_new
        return x


class CsiL2ONetwork(nn.Module):
    def __init__(self, M, Na=32, Nt=32, Ni=256, top_G=51, T_iterations=10):
        super().__init__()
        self.Na, self.Nt = Na, Nt; h_dim = 2 * Na * Nt
        self.encoder_W        = nn.Linear(h_dim, M, bias=False)
        self.sparse_transform = AngularSparseTransform(row_dim=2*Nt, Ni=Ni, top_G=top_G)
        self.l2o_module       = L2OModule(signal_dim=h_dim, T_iterations=T_iterations)

    def forward(self, h_vec):
        B = h_vec.shape[0]; s = self.encoder_W(h_vec)
        h_mat        = h_vec.view(B, self.Na, 2 * self.Nt)
        h_sparse_rec = self.sparse_transform(h_mat).reshape(B, -1)
        h_rec        = self.l2o_module(s, self.encoder_W.weight)
        return h_rec, h_sparse_rec


def make_sensing_matrix(M, W_trained):
    D = W_trained.shape[1]
    if M >= TRAINED_M:
        if M == TRAINED_M: return W_trained.clone()
        norm    = W_trained.norm('fro').item() / TRAINED_M
        W_extra = torch.randn(M - TRAINED_M, D) * (norm / math.sqrt(D))
        return torch.cat([W_trained.cpu(), W_extra], dim=0)
    U, S, Vt = torch.linalg.svd(W_trained.cpu(), full_matrices=False)
    return (S[:M].unsqueeze(1) * Vt[:M]).clone()


def build_model(M, W_sensing, ckpt_path):
    model      = CsiL2ONetwork(M=M, Na=Na, Nt=Nt, Ni=Ni, top_G=TOP_G, T_iterations=T_ITERS).to(device)
    saved      = torch.load(ckpt_path, map_location=device)
    model_dict = model.state_dict()
    for k, v in saved.items():
        if k.startswith('encoder_W'): continue
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v
    model.load_state_dict(model_dict)
    with torch.no_grad():
        model.encoder_W.weight.copy_(W_sensing.to(device))
    return model


def nmse_linear(model, x_test):
    loader = DataLoader(TensorDataset(x_test, x_test), batch_size=BATCH, shuffle=False)
    model.eval(); nmse_sum = 0.0; n = 0
    with torch.inference_mode():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            h_rec, _ = model(data)
            mse      = ((target - h_rec) ** 2).sum(dim=1)
            norm_sq  = (target ** 2).sum(dim=1)
            nmse_sum += (mse / norm_sq).sum().item()
            n        += data.shape[0]
    return nmse_sum / n   # linear (not dB)


actual_proposed = None

if PLOT_MODE == 'hybrid':
    print("Loading QuaDRiGa data for actual model evaluation...")
    try:
        with h5py.File(QUADRIGA_PATH, 'r') as f:
            keys     = list(f.keys())
            data_key = 'HT' if 'HT' in keys else [k for k in keys if not k.startswith('#')][0]
            raw5d    = np.array(f[data_key])
    except Exception as e:
        mat      = sio.loadmat(QUADRIGA_PATH)
        keys     = [k for k in mat.keys() if not k.startswith('__')]
        data_key = 'HT' if 'HT' in keys else keys[-1]
        raw5d    = mat[data_key]

    Rx_f, Tx_f, RI, Sub, N = raw5d.shape

    # Best single-subcarrier crop (subcarrier 0, first 32×32 antennas)
    sl    = raw5d[:Na, :Nt, :, 0, :].transpose(3, 2, 0, 1).reshape(N, -1)
    x_raw = torch.tensor(sl.astype(np.float32))

    # Per-sample L2-normalisation (most robust without knowing COST2100 stats)
    norms = x_raw.norm(dim=1, keepdim=True).clamp(min=1e-12)
    x_test = x_raw / norms

    saved_ckpt = torch.load(MODEL_PATH, map_location='cpu')
    W_trained  = saved_ckpt['encoder_W.weight']

    print("Running model across all CRs...")
    actual_proposed = []
    for M in CR_M:
        W_s   = make_sensing_matrix(M, W_trained)
        model = build_model(M, W_s, MODEL_PATH)
        val   = nmse_linear(model, x_test)
        actual_proposed.append(val)
        print(f"  CR=H/{H_DIM//M:3d}  M={M:4d}  NMSE={10*math.log10(max(val,1e-10)):+.2f} dB  "
              f"(linear={val:.4f})")

print("\nGenerating Figure 7 plot...")

METHODS = {
    'ISTA':     dict(color='black',    marker='+',  ls='-',  lw=1.4, ms=8,  label='ISTA'),
    'MS4L2O':   dict(color='#D95319', marker='d',  ls='-',  lw=1.4, ms=7,  label='MS4L2O'),
    'CsiNet':   dict(color='#EDB120', marker='*',  ls='-',  lw=1.4, ms=9,  label='CsiNet'),
    'TiLISTA':  dict(color='#7E2F8E', marker='o',  ls='-',  lw=1.4, ms=7,  mfc='none', label='TiLISTA'),
    'TransNet': dict(color='#77AC30', marker='^',  ls='-',  lw=1.4, ms=7,  mfc='none', label='TransNet'),
    'Proposed': dict(color='#0072BD', marker='x',  ls='-',  lw=1.8, ms=9,  label='Proposed'),
}

x_pos = np.arange(len(CR_LABELS))   # 0,1,2,3,4

fig, ax = plt.subplots(figsize=(6.5, 5.2))

for name, style in METHODS.items():
    y_vals = PAPER_NMSE_LINEAR[name]

    if PLOT_MODE == 'hybrid' and name == 'Proposed' and actual_proposed is not None:
        y_vals = actual_proposed
        style  = dict(style)          # copy so we can modify
        style['label'] = 'Proposed (COST2100 weights)'
        style['ls']    = '--'

    kw = {k: v for k, v in style.items() if k != 'label'}
    ax.semilogy(x_pos, y_vals, **kw, label=style['label'])

    if PLOT_MODE == 'hybrid' and name == 'Proposed' and actual_proposed is not None:
        ax.semilogy(x_pos, PAPER_NMSE_LINEAR['Proposed'],
                    color='#0072BD', marker='x', ls=':', lw=1.2, ms=9, alpha=0.5,
                    label='Proposed (paper, QuaDRiGa-trained)')

ax.set_xticks(x_pos)
ax.set_xticklabels(CR_LABELS, fontsize=11)
ax.set_xlabel('Compression Ratio', fontsize=12)
ax.set_ylabel('NMSE', fontsize=12)

ax.set_ylim(0.06, 1.5)
ax.set_xlim(-0.3, len(CR_LABELS) - 0.7)
ax.yaxis.grid(True, which='both', linestyle='--', linewidth=0.6, color='#AAAAAA')
ax.set_axisbelow(True)
ax.xaxis.grid(False)

ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax.yaxis.set_minor_locator(ticker.LogLocator(subs=np.arange(2, 10) * 0.1))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

ax.tick_params(axis='both', which='major', labelsize=10)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

legend = ax.legend(loc='lower left', fontsize=9.5, framealpha=0.92,
                   edgecolor='#CCCCCC', ncol=1)

fig.tight_layout(pad=0.8)

out_path = '/kaggle/working/figure7_quadriga.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\nSaved → {out_path}")

print(f"\n{'='*65}")
print(f"  Figure 7 values  (linear NMSE — same as paper y-axis)")
print(f"{'='*65}")
header = f"  {'CR':<6}" + "".join(f"  {m:>12}" for m in METHODS)
print(header)
print(f"  {'-'*63}")
for i, cr in enumerate(CR_LABELS):
    row = f"  {cr:<6}"
    for name in METHODS:
        vals = actual_proposed if (PLOT_MODE == 'hybrid' and
                                   name == 'Proposed' and
                                   actual_proposed is not None) \
               else PAPER_NMSE_LINEAR[name]
        row += f"  {vals[i]:>12.4f}"
    print(row)
print(f"{'='*65}")

if PLOT_MODE == 'hybrid':
    print("\n⚠  HYBRID MODE NOTE:")
    print("   The dashed blue line shows your COST2100-trained model on QuaDRiGa.")
    print("   To match the paper's solid blue line, train on QuaDRiGa data:")
    print("   - Use the same architecture (Na=Nt=32, crop from 64×64 grid)")
    print("   - Train on QuaDRiGa train split with the same loss (Eq. 11, β=0.01)")
    print("   - The encoder W will adapt to QuaDRiGa's channel statistics automatically.")
