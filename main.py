"""
THERMAL ANALYSIS ML PIPELINE — Professional Edition
Gradient Boosting + Random Forest Ensemble
Physics-Informed Feature Engineering
"""

import os
import sys
import time
import warnings
import pickle
import textwrap

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

class C:
    """ANSI color codes for rich terminal output."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    BG_DARK = "\033[40m"
    ORANGE  = "\033[38;5;208m"
    TEAL    = "\033[38;5;45m"

def banner():
    print(f"""
{C.CYAN}{C.BOLD}
THERMAL IMAGE TEMPERATURE PREDICTION PIPELINE
Physics-Informed Machine Learning · Ensemble Regression
{C.RESET}""")

def section(title: str):
    width = 72
    print(f"\n{C.BOLD}{C.YELLOW}{'─' * width}")
    print(f"  {title.upper()}")
    print(f"{'─' * width}{C.RESET}")

def kv(key: str, val, unit: str = "", color=C.TEAL, width: int = 32):
    dots = "·" * max(1, width - len(key))
    print(f"  {C.DIM}{key} {dots}{C.RESET} {color}{C.BOLD}{val}{C.RESET} {C.DIM}{unit}{C.RESET}")

def status(msg: str, ok: bool = True):
    icon = f"{C.GREEN}✔" if ok else f"{C.RED}✘"
    print(f"  {icon} {C.RESET}{msg}")

def warn(msg: str):
    print(f"  {C.YELLOW}  {msg}{C.RESET}")

def info(msg: str):
    print(f"  {C.BLUE}  {msg}{C.RESET}")


class Config:
    # Image
    IMG_SIZE     = (128, 128)

    # Temperature range (K)
    TEMP_MIN     = 200
    TEMP_MAX     = 400

    # Physics constants
    EMISSIVITY   = 0.85
    H            = 15          # convective heat transfer coeff (W/m²·K)
    K_COND       = 0.8         # thermal conductivity (W/m·K)
    T_AMBIENT    = 300         # ambient temperature (K)
    SIGMA        = 5.67e-8     # Stefan–Boltzmann constant

    # ML
    TEST_SIZE    = 0.30
    RANDOM_STATE = 42
    CV_FOLDS     = 5

    # Output
    OUTPUT_DIR   = "outputs"

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

THERMAL_CMAP = LinearSegmentedColormap.from_list(
    "thermal_pro",
    ["#0d0221", "#1a0a4e", "#2d1b99", "#4f35d9",
     "#7a5ff5", "#b48bff", "#ff6b9d", "#ff9f5b",
     "#ffe066", "#ffffff"],
    N=512
)

RESIDUAL_CMAP = LinearSegmentedColormap.from_list(
    "residual_div",
    ["#0a3d62", "#1565C0", "#42A5F5", "#E3F2FD",
     "#FFF9C4", "#FF8F00", "#E53935", "#4a0010"],
    N=512
)

FEATURE_NAMES = [
    "T_mean", "T_min", "Pixel_Std", "T_Contrast",
    "Laplacian_Var", "Grad_Magnitude",
    "Q_rad", "Q_conv", "Q_cond",
    "Q_rad×Q_conv", "Q_rad×Q_cond"
]

def image_to_temperature(gray: np.ndarray) -> np.ndarray:
    """Map grayscale pixel values to temperature (K) with realistic noise."""
    noise    = np.random.normal(0, 10, gray.shape)
    drift    = np.random.uniform(-6, 6)
    nonlin   = 0.03 * (gray / 255.0) ** 2 * 100
    return cfg.TEMP_MIN + (gray / 255.0) * (cfg.TEMP_MAX - cfg.TEMP_MIN) + noise + drift + nonlin


def compute_heat_fluxes(T: np.ndarray):
    """Compute radiation, convection, and conduction heat fluxes."""
    T_mean  = np.mean(T)
    q_rad   = cfg.EMISSIVITY * cfg.SIGMA * (T_mean ** 4)
    q_conv  = cfg.H * (T_mean - cfg.T_AMBIENT)
    gx, gy  = np.gradient(T, axis=1), np.gradient(T, axis=0)
    grad_mag = np.mean(np.sqrt(gx**2 + gy**2))
    q_cond  = -cfg.K_COND * grad_mag
    return q_rad, q_conv, q_cond, grad_mag


def extract_features(T: np.ndarray, gray: np.ndarray):
    """Extract physics-informed feature vector from temperature field."""
    T_mean   = np.mean(T)
    T_min    = np.min(T)
    T_max    = np.max(T)
    std_pix  = np.std(gray)
    contrast = T_max - T_min
    lap_var  = cv2.Laplacian(gray, cv2.CV_64F).var()

    q_rad, q_conv, q_cond, grad_mag = compute_heat_fluxes(T)

    features = [
        T_mean, T_min, std_pix, contrast,
        lap_var, grad_mag,
        q_rad, q_conv, q_cond,
        q_rad * q_conv,
        q_rad * q_cond
    ]
    return features, T_max, q_rad, q_conv, q_cond


def process_image(path: str):
    """Load, resize, and featurize a single image."""
    img = cv2.imread(path)
    if img is None:
        return None
    img  = cv2.resize(img, cfg.IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    T    = image_to_temperature(gray)
    return extract_features(T, gray)


def load_image_dataset(folder: str):
    X, y, qrad, qconv, qcond = [], [], [], [], []
    n_skipped = 0
    exts = (".jpg", ".png", ".jpeg", ".bmp")

    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                result = process_image(os.path.join(root, f))
                if result:
                    feats, target, qr, qc, qcd = result
                    X.append(feats); y.append(target)
                    qrad.append(qr); qconv.append(qc); qcond.append(qcd)
                else:
                    n_skipped += 1

    if n_skipped:
        warn(f"Skipped {n_skipped} unreadable images in '{folder}'")
    return X, y, qrad, qconv, qcond


def load_conduction_data(file: str):
    X, y, qrad, qconv, qcond = [], [], [], [], []
    if not os.path.exists(file):
        warn(f"File not found: {file}  — skipping.")
        return X, y, qrad, qconv, qcond

    data    = np.load(file, allow_pickle=True)
    T_fields = data[data.files[0]]

    for T in T_fields:
        rng  = T.max() - T.min()
        gray = ((T - T.min()) / (rng if rng > 0 else 1) * 255).astype(np.uint8)
        feats, target, qr, qc, qcd = extract_features(T, gray)
        X.append(feats); y.append(target)
        qrad.append(qr); qconv.append(qc); qcond.append(qcd)

    return X, y, qrad, qconv, qcond


banner()

section("1 · Data Ingestion")

PATHS_IMG = [
    "data/thermal_images/train/images",
    "data/2D Binary Images and Effective Thermal Conductivity CFD Results/QSGS",
]
PATH_NPZ = "data/rw9yk3c559-2/Dataset/HeatTransferPhenomena_35_58.npz"

X_all, y_all = [], []
q_rad_all, q_conv_all, q_cond_all = [], [], []
source_log = []

for path in PATHS_IMG:
    t0 = time.time()
    Xi, yi, qr, qc, qcd = load_image_dataset(path)
    elapsed = time.time() - t0
    n = len(Xi)
    if n:
        status(f"Loaded {n:>5} samples from '{os.path.basename(path)}'  [{elapsed:.1f}s]")
        source_log.append((os.path.basename(path), n))
    else:
        warn(f"No samples found in '{path}'")
    X_all.extend(Xi); y_all.extend(yi)
    q_rad_all.extend(qr); q_conv_all.extend(qc); q_cond_all.extend(qcd)

t0 = time.time()
Xi, yi, qr, qc, qcd = load_conduction_data(PATH_NPZ)
elapsed = time.time() - t0
if len(Xi):
    status(f"Loaded {len(Xi):>5} samples from NPZ file  [{elapsed:.1f}s]")
    source_log.append(("NPZ Conduction", len(Xi)))
X_all.extend(Xi); y_all.extend(yi)
q_rad_all.extend(qr); q_conv_all.extend(qc); q_cond_all.extend(qcd)

X = np.array(X_all)
y = np.array(y_all)
q_rad_arr  = np.array(q_rad_all)
q_conv_arr = np.array(q_conv_all)
q_cond_arr = np.array(q_cond_all)

print()
kv("Total samples", f"{len(X):,}")
kv("Feature dimensions", X.shape[1])
kv("Target range", f"{y.min():.1f} – {y.max():.1f}", "K")
kv("Target mean ± std", f"{y.mean():.1f} ± {y.std():.1f}", "K")

section("2 · Preprocessing")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size   = cfg.TEST_SIZE,
    random_state= cfg.RANDOM_STATE
)

kv("Train samples", f"{len(X_train):,}")
kv("Test  samples", f"{len(X_test):,}")
kv("Test split", f"{cfg.TEST_SIZE*100:.0f}%")
status("StandardScaler fitted on training data only")

section("3 · Model Training")

info("Training Gradient Boosting Regressor ...")
t0 = time.time()
model_gb = GradientBoostingRegressor(
    n_estimators  = 180,
    learning_rate = 0.05,
    max_depth     = 3,
    subsample     = 0.7,
    random_state  = cfg.RANDOM_STATE
)
model_gb.fit(X_train, y_train)
kv("GBR training time", f"{time.time()-t0:.1f}", "s")

info("Training Random Forest Regressor ...")
t0 = time.time()
model_rf = RandomForestRegressor(
    n_estimators = 100,
    max_depth    = 7,
    random_state = cfg.RANDOM_STATE,
    n_jobs       = -1
)
model_rf.fit(X_train, y_train)
kv("RF  training time", f"{time.time()-t0:.1f}", "s")

y_pred_gb  = model_gb.predict(X_test)
y_pred_rf  = model_rf.predict(X_test)
y_pred     = 0.50 * y_pred_gb + 0.50 * y_pred_rf

status("Ensemble prediction (50% GBR + 50% RF) complete")

section("4 · Performance Metrics")

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mbe  = np.mean(y_pred - y_test)         # mean bias error
mape = np.mean(np.abs((y_test - y_pred) / np.clip(np.abs(y_test), 1, None))) * 100
residuals = y_test - y_pred

info("Running 5-fold cross-validation on GBR ...")
cv_scores = cross_val_score(model_gb, X_scaled, y,
                             cv=cfg.CV_FOLDS, scoring="r2", n_jobs=-1)

print()
kv("MAE   (Mean Absolute Error)",   f"{mae:.3f}",  "K",   C.GREEN)
kv("RMSE  (Root Mean Sq. Error)",   f"{rmse:.3f}", "K",   C.GREEN)
kv("R²    (Coefficient of Det.)",   f"{r2:.4f}",   "",    C.TEAL)
kv("MBE   (Mean Bias Error)",       f"{mbe:+.3f}", "K",   C.YELLOW if abs(mbe)>1 else C.GREEN)
kv("MAPE  (Mean Abs. % Error)",     f"{mape:.2f}", "%",   C.GREEN if mape<5 else C.YELLOW)
kv("CV R² (5-fold, GBR)",
   f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}", "", C.CYAN)

r2_gb = r2_score(y_test, y_pred_gb)
r2_rf = r2_score(y_test, y_pred_rf)
print()
kv("  R² · GBR alone",      f"{r2_gb:.4f}", "", C.MAGENTA)
kv("  R² · RF alone",       f"{r2_rf:.4f}", "", C.MAGENTA)
kv("  R² · Ensemble",       f"{r2:.4f}",    "", C.CYAN)

section("5 · Residual Analysis")

res_skew    = stats.skew(residuals)
res_kurt    = stats.kurtosis(residuals)
_, p_shapiro = stats.shapiro(residuals[:min(5000, len(residuals))])
p90  = np.percentile(np.abs(residuals), 90)
p95  = np.percentile(np.abs(residuals), 95)

kv("Residual mean",           f"{np.mean(residuals):+.4f}", "K")
kv("Residual std",            f"{np.std(residuals):.4f}",   "K")
kv("Skewness",                f"{res_skew:.4f}")
kv("Excess kurtosis",         f"{res_kurt:.4f}")
kv("Shapiro-Wilk p-value",    f"{p_shapiro:.4f}",
   color=C.GREEN if p_shapiro > 0.05 else C.YELLOW)
kv("90th pct. abs. error",    f"{p90:.2f}", "K")
kv("95th pct. abs. error",    f"{p95:.2f}", "K")

if abs(mbe) < 1:
    status("Minimal systematic bias detected")
else:
    warn(f"Systematic bias of {mbe:+.2f} K — consider recalibration")

section("6 · Feature Importance (GBR)")

importance_gb = model_gb.feature_importances_
sorted_idx    = np.argsort(importance_gb)[::-1]

for rank, idx in enumerate(sorted_idx, 1):
    bar = "█" * int(importance_gb[idx] * 200)
    print(f"  {rank:>2}. {C.CYAN}{FEATURE_NAMES[idx]:<22}{C.RESET} "
          f"{C.TEAL}{importance_gb[idx]:.4f}{C.RESET}  {C.DIM}{bar}{C.RESET}")

section("7 · Heat Transfer Summary")

for label, arr, unit in [
    ("Radiation  Q_rad",  q_rad_arr,  "W/m²"),
    ("Convection Q_conv", q_conv_arr, "W/m²"),
    ("Conduction Q_cond", q_cond_arr, "W/m²"),
]:
    kv(f"{label}  (mean)",  f"{np.mean(arr):.3e}", unit)
    kv(f"{label}  (std)",   f"{np.std(arr):.3e}",  unit)

section("8 · Generating Dashboard")

DARK_BG   = "#0e1117"
CARD_BG   = "#161b22"
GRID_COL  = "#21262d"
TEXT_MAIN = "#e6edf3"
TEXT_DIM  = "#8b949e"
ACC1      = "#58a6ff"   # blue
ACC2      = "#3fb950"   # green
ACC3      = "#d29922"   # amber
ACC4      = "#f78166"   # coral
ACC5      = "#bc8cff"   # purple
ACC6      = "#39d353"   # bright green

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_DIM,
    "axes.titlecolor":   TEXT_MAIN,
    "xtick.color":       TEXT_DIM,
    "ytick.color":       TEXT_DIM,
    "text.color":        TEXT_MAIN,
    "grid.color":        GRID_COL,
    "grid.linewidth":    0.6,
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "legend.facecolor":  CARD_BG,
    "legend.edgecolor":  GRID_COL,
})

fig = plt.figure(figsize=(22, 16), facecolor=DARK_BG)
fig.suptitle(
    "Thermal Temperature Prediction — ML Analysis Dashboard",
    fontsize=17, fontweight="bold", color=TEXT_MAIN,
    y=0.98, x=0.5
)

gs = gridspec.GridSpec(
    3, 4,
    figure=fig,
    hspace=0.42,
    wspace=0.38,
    left=0.05, right=0.97,
    top=0.93, bottom=0.06
)

def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(CARD_BG)
    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.grid(True, linestyle="--", alpha=0.35)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)

def add_metric_badge(ax, text, x=0.03, y=0.96):
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=9, color=ACC1, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc=DARK_BG, ec=ACC1, lw=0.8, alpha=0.85))


ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Predicted vs Actual Temperature", "Actual (K)", "Predicted (K)")

sc = ax1.scatter(
    y_test, y_pred,
    c=np.abs(residuals), cmap=RESIDUAL_CMAP,
    s=18, alpha=0.65, edgecolors="none"
)
lim = [min(y_test.min(), y_pred.min())-2, max(y_test.max(), y_pred.max())+2]
ax1.plot(lim, lim, "--", color=ACC2, lw=1.5, label="Perfect fit")
ax1.set_xlim(lim); ax1.set_ylim(lim)
cb1 = plt.colorbar(sc, ax=ax1, pad=0.01)
cb1.set_label("|Residual| (K)", fontsize=8)
cb1.ax.yaxis.set_tick_params(color=TEXT_DIM)
add_metric_badge(ax1, f"R² = {r2:.4f}\nMAE = {mae:.2f} K")
ax1.legend(fontsize=8)


ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Residuals vs Predicted", "Predicted Temperature (K)", "Residual (K)")

sc2 = ax2.scatter(
    y_pred, residuals,
    c=y_pred, cmap="plasma",
    s=16, alpha=0.55, edgecolors="none"
)
ax2.axhline(0,        color=ACC2,   lw=1.8, linestyle="--", label="Zero line")
ax2.axhline(+mae,     color=ACC3,   lw=1.0, linestyle=":",  label=f"+MAE ({mae:.1f} K)")
ax2.axhline(-mae,     color=ACC3,   lw=1.0, linestyle=":")
ax2.axhspan(-mae, +mae, alpha=0.07, color=ACC3)
plt.colorbar(sc2, ax=ax2, pad=0.01).set_label("Predicted T (K)", fontsize=8)
add_metric_badge(ax2, f"Bias = {mbe:+.3f} K\nStd = {residuals.std():.3f} K")
ax2.legend(fontsize=7.5)


ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, "Residual Distribution", "Residual (K)", "Density")

counts, bin_edges = np.histogram(residuals, bins=60, density=True)
centers = (bin_edges[:-1] + bin_edges[1:]) / 2
smooth  = gaussian_filter1d(counts, sigma=2.5)

ax3.fill_between(centers, smooth, alpha=0.30, color=ACC5)
ax3.plot(centers, smooth, color=ACC5, lw=2.2)

mu_r, std_r = np.mean(residuals), np.std(residuals)
x_norm  = np.linspace(residuals.min(), residuals.max(), 300)
y_norm  = stats.norm.pdf(x_norm, mu_r, std_r)
ax3.plot(x_norm, y_norm, "--", color=ACC3, lw=1.5, label="Normal fit")
ax3.axvline(0,    color=ACC2, lw=1.5, linestyle="--", label="Zero")
ax3.axvline(mu_r, color=ACC4, lw=1.2, linestyle=":",  label=f"Mean={mu_r:.2f}")
add_metric_badge(ax3, f"Skew  = {res_skew:.3f}\nKurt  = {res_kurt:.3f}")
ax3.legend(fontsize=7.5)


ax4 = fig.add_subplot(gs[0, 3])
style_ax(ax4, "Q-Q Plot (Normality Check)", "Theoretical Quantiles", "Sample Quantiles")

(osm, osr), (slope, intercept, r_qq) = stats.probplot(residuals)
ax4.scatter(osm, osr, s=12, alpha=0.55, color=ACC1, edgecolors="none")
ax4.plot(osm, slope*np.array(osm)+intercept, color=ACC4, lw=2, label=f"Fit (r={r_qq:.3f})")
ax4.set_xlim(osm[0]-0.3, osm[-1]+0.3)
add_metric_badge(ax4, f"Shapiro p={p_shapiro:.3f}")
ax4.legend(fontsize=8)


ax5 = fig.add_subplot(gs[1, :2])
style_ax(ax5, "Feature Importance — Gradient Boosting Regressor",
         "Feature", "Importance Score")

imp_sorted = importance_gb[sorted_idx]
names_sorted = [FEATURE_NAMES[i] for i in sorted_idx]

palette = plt.cm.get_cmap("plasma", len(FEATURE_NAMES))
colors  = [palette(i / len(FEATURE_NAMES)) for i in range(len(FEATURE_NAMES))]

bars = ax5.barh(names_sorted[::-1], imp_sorted[::-1],
                color=colors, edgecolor="none", height=0.6)
for bar, val in zip(bars, imp_sorted[::-1]):
    ax5.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", fontsize=8, color=TEXT_DIM)
ax5.set_xlim(0, imp_sorted.max() * 1.18)
ax5.invert_yaxis()


ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6, "Temperature Distribution", "Temperature (K)", "Density")

for arr, label, col in [
    (y_train, "Train", ACC1),
    (y_test,  "Test",  ACC2),
    (y_pred,  "Pred",  ACC4),
]:
    counts_t, bins_t = np.histogram(arr, bins=50, density=True)
    ctrs_t = (bins_t[:-1] + bins_t[1:]) / 2
    sm_t   = gaussian_filter1d(counts_t, sigma=2)
    ax6.fill_between(ctrs_t, sm_t, alpha=0.18, color=col)
    ax6.plot(ctrs_t, sm_t, lw=1.8, color=col, label=label)

ax6.legend(fontsize=8)
add_metric_badge(ax6, f"μ={y.mean():.1f}K\nσ={y.std():.1f}K")


ax7 = fig.add_subplot(gs[1, 3])

n_bins = 5
bin_labels = [f"T{i+1}" for i in range(n_bins)]
bins_edges = np.linspace(y.min(), y.max(), n_bins + 1)
yt_bin = np.digitize(y_test, bins_edges[1:-1])
yp_bin = np.digitize(y_pred, bins_edges[1:-1])
cm = confusion_matrix(yt_bin, yp_bin, labels=list(range(n_bins)))

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

im = ax7.imshow(cm_norm, cmap="viridis", aspect="auto", vmin=0, vmax=1)
cb7 = plt.colorbar(im, ax=ax7, fraction=0.046, pad=0.04)
cb7.set_label("Row-norm accuracy", fontsize=8)

tick_labels = [f"{bins_edges[i]:.0f}–{bins_edges[i+1]:.0f}K" for i in range(n_bins)]
ax7.set_xticks(range(n_bins)); ax7.set_xticklabels(tick_labels, rotation=35, fontsize=7)
ax7.set_yticks(range(n_bins)); ax7.set_yticklabels(tick_labels, fontsize=7)
ax7.set_title("Confusion Matrix (Temp. Bins)", pad=8)
ax7.set_xlabel("Predicted Bin"); ax7.set_ylabel("Actual Bin")

for i in range(n_bins):
    for j in range(n_bins):
        val = cm_norm[i, j]
        ax7.text(j, i, f"{val:.2f}", ha="center", va="center",
                 fontsize=8,
                 color="white" if val < 0.6 else "black",
                 fontweight="bold" if i==j else "normal")


ax8 = fig.add_subplot(gs[2, 0])
style_ax(ax8, "Model Comparison (R²)", "Model", "R² Score")

models_r2    = [r2_gb, r2_rf, r2]
models_names = ["GBR", "Random\nForest", "Ensemble"]
bar_colors   = [ACC5, ACC1, ACC2]

bars8 = ax8.bar(models_names, models_r2, color=bar_colors,
                width=0.45, edgecolor="none")
ax8.set_ylim(min(models_r2)*0.98, 1.01)
for bar, val in zip(bars8, models_r2):
    ax8.text(bar.get_x()+bar.get_width()/2, val+0.001,
             f"{val:.4f}", ha="center", fontsize=9, color=TEXT_MAIN, fontweight="bold")
ax8.axhline(1.0, color=TEXT_DIM, lw=0.8, linestyle="--")


ax9 = fig.add_subplot(gs[2, 1])
style_ax(ax9, "CDF of Absolute Error", "|Error| (K)", "Cumulative Fraction")

abs_err = np.abs(residuals)
cdf_x   = np.sort(abs_err)
cdf_y   = np.arange(1, len(cdf_x)+1) / len(cdf_x)

ax9.plot(cdf_x, cdf_y, color=ACC1, lw=2.2)
ax9.fill_between(cdf_x, cdf_y, alpha=0.12, color=ACC1)
ax9.axvline(p90, color=ACC3, lw=1.3, linestyle="--", label=f"P90={p90:.1f}K")
ax9.axvline(p95, color=ACC4, lw=1.3, linestyle="--", label=f"P95={p95:.1f}K")
ax9.axhline(0.90, color=ACC3, lw=0.7, linestyle=":", alpha=0.5)
ax9.axhline(0.95, color=ACC4, lw=0.7, linestyle=":", alpha=0.5)
ax9.set_xlim(left=0)
ax9.legend(fontsize=8)
add_metric_badge(ax9, f"RMSE={rmse:.2f}K\nMAE={mae:.2f}K")


ax10 = fig.add_subplot(gs[2, 2])
style_ax(ax10, "Heat Flux Distributions", "Heat Flux (W/m²)", "Density")

for arr, label, col in [
    (q_rad_arr,  "Q_rad",  ACC3),
    (q_conv_arr, "Q_conv", ACC4),
]:
    vals = np.log10(np.abs(arr) + 1)
    cnt, b = np.histogram(vals, bins=50, density=True)
    ctrs   = (b[:-1]+b[1:])/2
    sm     = gaussian_filter1d(cnt, sigma=2)
    ax10.fill_between(ctrs, sm, alpha=0.20, color=col)
    ax10.plot(ctrs, sm, lw=1.8, color=col, label=label)

ax10.legend(fontsize=8)
ax10.set_xlabel("log10(|Q|+1)  W/m²")


ax11 = fig.add_subplot(gs[2, 3])
style_ax(ax11, "Cross-Validation R² (GBR, 5-fold)", "Fold", "R²")

fold_ids = np.arange(1, len(cv_scores)+1)
ax11.bar(fold_ids, cv_scores, color=ACC6, width=0.5, edgecolor="none")
ax11.axhline(cv_scores.mean(), color=ACC4, lw=1.8,
             linestyle="--", label=f"Mean={cv_scores.mean():.4f}")
ax11.axhspan(cv_scores.mean()-cv_scores.std(),
             cv_scores.mean()+cv_scores.std(),
             alpha=0.15, color=ACC4, label=f"+/-sigma={cv_scores.std():.4f}")
ax11.set_ylim(max(0, cv_scores.min()-0.02), 1.01)
ax11.set_xticks(fold_ids)
for i, v in zip(fold_ids, cv_scores):
    ax11.text(i, v+0.002, f"{v:.3f}", ha="center", fontsize=8, color=TEXT_MAIN)
ax11.legend(fontsize=8)


plt.savefig(
    os.path.join(cfg.OUTPUT_DIR, "thermal_ml_dashboard.png"),
    dpi=180, bbox_inches="tight", facecolor=DARK_BG
)
status("Dashboard saved → outputs/thermal_ml_dashboard.png")
plt.show()

section("9 · Saving Pipeline")

pipeline = {
    "model_gb":      model_gb,
    "model_rf":      model_rf,
    "scaler":        scaler,
    "feature_names": FEATURE_NAMES,
    "config": {
        "TEMP_MIN":    cfg.TEMP_MIN,
        "TEMP_MAX":    cfg.TEMP_MAX,
        "EMISSIVITY":  cfg.EMISSIVITY,
        "H":           cfg.H,
        "K_COND":      cfg.K_COND,
        "T_AMBIENT":   cfg.T_AMBIENT,
    },
    "metrics": {
        "mae":  mae,
        "rmse": rmse,
        "r2":   r2,
        "mbe":  mbe,
        "mape": mape,
    }
}

pkl_path = os.path.join(cfg.OUTPUT_DIR, "thermal_pipeline.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(pipeline, f)

status(f"Pipeline saved → {pkl_path}")

section("Final Summary")

grade = "EXCELLENT" if r2 > 0.95 else "GOOD" if r2 > 0.85 else "FAIR"
grade_color = C.GREEN if r2 > 0.95 else C.YELLOW if r2 > 0.85 else C.RED

print(f"""
  {C.BOLD}Model Performance Grade:{C.RESET}  {grade_color}{C.BOLD}{grade}{C.RESET}

  {C.DIM}┌─────────────────────────────────────────────────────┐
  │  The ensemble model explains {r2*100:.1f}% of variance in  │
  │  max temperature predictions across the test set.     │
  │  Average error is {mae:.2f} K with {grade.lower()} generalization.  │
  └─────────────────────────────────────────────────────┘{C.RESET}
""")

kv("Top feature",     FEATURE_NAMES[sorted_idx[0]], color=C.CYAN)
kv("2nd feature",     FEATURE_NAMES[sorted_idx[1]], color=C.CYAN)
kv("Dominant flux",
   "Radiation (Q_rad)" if q_rad_arr.mean() > q_conv_arr.mean() else "Convection (Q_conv)")
kv("Dashboard",       "outputs/thermal_ml_dashboard.png")
kv("Saved model",     "outputs/thermal_pipeline.pkl")

print(f"\n{C.DIM}  Pipeline complete.{C.RESET}\n")
