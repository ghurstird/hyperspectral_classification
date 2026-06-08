"""
HSI Classification GUI — PyQt6
Tek dosya: gui/app.py  |  python3 gui/app.py
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
import os, sys
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score
from scipy.ndimage import gaussian_filter as scipy_gauss

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QSizePolicy, QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

# ──────────────────────────────────────────────────────────────────────────────
# Section 1 — Global stylesheet  (applied once on QApplication)
# ──────────────────────────────────────────────────────────────────────────────

APP_QSS = """
/* ── Window & base ─────────────────────────────────────────────────────── */
QMainWindow, QScrollArea, QWidget#scroll_content {
    background: #f1f5f9;
}

/* ── Labels: always dark on transparent bg ─────────────────────────────── */
QLabel {
    color: #0f172a;
    background: transparent;
    border: none;
}

/* ── White card frame ───────────────────────────────────────────────────── */
QFrame#card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
}
QFrame#card QLabel {
    color: #0f172a;
    background: transparent;
    border: none;
}
QFrame#card_accent_blue {
    background: white;
    border: 2px solid #6366f1;
    border-radius: 12px;
}
QFrame#card_accent_green {
    background: white;
    border: 2px solid #10b981;
    border-radius: 12px;
}
QFrame#card_accent_red {
    background: white;
    border: 2px solid #ef4444;
    border-radius: 12px;
}
QFrame#card_accent_blue QLabel,
QFrame#card_accent_green QLabel,
QFrame#card_accent_red QLabel {
    color: #0f172a;
    background: transparent;
    border: none;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
QPushButton#primary_btn {
    background: #7c3aed;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 11px;
    font-weight: bold;
    font-size: 14px;
}
QPushButton#primary_btn:hover   { background: #6d28d9; }
QPushButton#primary_btn:pressed { background: #5b21b6; }
QPushButton#primary_btn:disabled { background: #cbd5e1; color: #94a3b8; }

QPushButton#ghost_btn {
    background: white;
    color: #374151;
    border: 1px solid #e2e8f0;
    border-radius: 7px;
    padding: 7px 12px;
    font-size: 12px;
}
QPushButton#ghost_btn:hover   { background: #f8fafc; border-color: #7c3aed; color: #7c3aed; }
QPushButton#ghost_btn:disabled { background: #f8fafc; color: #94a3b8; }

/* ── Progress bar ───────────────────────────────────────────────────────── */
QProgressBar {
    border: none;
    border-radius: 5px;
    background: #e2e8f0;
    text-align: center;
    font-size: 11px;
    color: #374151;
    height: 10px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #7c3aed,stop:1 #6366f1);
    border-radius: 5px;
}

/* ── Table ─────────────────────────────────────────────────────────────── */
QTableWidget {
    background: white;
    alternate-background-color: #f8fafc;
    gridline-color: #f1f5f9;
    border: none;
    font-size: 13px;
    color: #0f172a;
    outline: none;
    selection-background-color: #ede9fe;
    selection-color: #1e1b4b;
}
QTableWidget::item {
    padding: 10px 14px;
    border: none;
    color: #0f172a;
}
QTableWidget::item:selected {
    background: #ede9fe;
    color: #1e1b4b;
}
QHeaderView {
    background: transparent;
}
QHeaderView::section {
    background: #f8fafc;
    color: #374151;
    padding: 11px 14px;
    border: none;
    border-bottom: 2px solid #e2e8f0;
    font-weight: bold;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Scrollbars ─────────────────────────────────────────────────────────── */
QScrollBar:vertical {
    background: #f1f5f9;
    width: 8px;
    border: none;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #cbd5e1;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: #94a3b8; }
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical { height: 0; }

QScrollBar:horizontal {
    background: #f1f5f9;
    height: 8px;
    border: none;
}
QScrollBar::handle:horizontal {
    background: #cbd5e1;
    border-radius: 4px;
}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal { width: 0; }
"""

# ──────────────────────────────────────────────────────────────────────────────
# Section 2 — Dataset & Model config
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR    = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "datasets", "processed")

DATASETS = {
    "indian_pines": {
        "display":     "Indian Pines",
        "prefix":      "indian_pines_gauss0.5_pca30",
        "gt_file":     "indian_pines_gt.npy",
        "num_classes": 16,
        "class_names": [
            "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
            "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
            "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean",
            "Wheat", "Woods", "Buildings-Grass-Trees", "Stone-Steel-Towers",
        ],
    },
    "pavia_university": {
        "display":     "Pavia University",
        "prefix":      "pavia_university_gauss0.5_pca30",
        "gt_file":     "pavia_university_gt.npy",
        "num_classes": 9,
        "class_names": [
            "Asphalt", "Meadows", "Gravel", "Trees", "Painted Metal Sheets",
            "Bare Soil", "Bitumen", "Self-Blocking Bricks", "Shadows",
        ],
    },
    "salinas": {
        "display":     "Salinas",
        "prefix":      "salinas_gauss0.5_pca30",
        "gt_file":     "salinas_gt.npy",
        "num_classes": 16,
        "class_names": [
            "Brocoli weeds 1", "Brocoli weeds 2", "Fallow", "Fallow rough plow",
            "Fallow smooth", "Stubble", "Celery", "Grapes untrained",
            "Soil vinyard", "Corn senesced", "Lettuce 4wk", "Lettuce 5wk",
            "Lettuce 6wk", "Lettuce 7wk", "Vinyard untrained", "Vinyard vertical",
        ],
    },
}

MODEL_CONFIGS = [
    {"key": "cnn2d_sup",  "label": "CNN2D", "arch": "cnn2d", "mode": "supervised",
     "file_prefix": "cnn_model"},
    {"key": "cnn3d_sup",  "label": "CNN3D", "arch": "cnn3d", "mode": "supervised",
     "file_prefix": "cnn3d_model"},
    {"key": "vit_sup",    "label": "ViT",   "arch": "vit",   "mode": "supervised",
     "file_prefix": "vit_model"},
    {"key": "cnn2d_semi", "label": "CNN2D", "arch": "cnn2d", "mode": "semi-supervised",
     "file_prefix": "semi_cnn_model"},
    {"key": "cnn3d_semi", "label": "CNN3D", "arch": "cnn3d", "mode": "semi-supervised",
     "file_prefix": "semi_cnn3d_model"},
    {"key": "vit_semi",   "label": "ViT",   "arch": "vit",   "mode": "semi-supervised",
     "file_prefix": "semi_vit_model"},
    {"key": "cnn2d_fed",  "label": "CNN2D", "arch": "cnn2d", "mode": "federated",
     "file_prefix": "federated_cnn_model"},
    {"key": "cnn3d_fed",  "label": "CNN3D", "arch": "cnn3d", "mode": "federated",
     "file_prefix": "federated_cnn3d_model"},
    {"key": "vit_fed",    "label": "ViT",   "arch": "vit",   "mode": "federated",
     "file_prefix": "federated_vit_model"},
]

_TOTAL_MODELS = len(MODEL_CONFIGS)  # 9

_CMAP_COLORS = [
    "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231",
    "#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4",
    "#469990","#dcbeff","#9A6324","#fffac8","#800000",
    "#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9",
]

def _make_cmap(n: int) -> ListedColormap:
    return ListedColormap(_CMAP_COLORS[:n])

# ──────────────────────────────────────────────────────────────────────────────
# Section 3 — Model architectures
# ──────────────────────────────────────────────────────────────────────────────

class HSI_CNN(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 30):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc_block = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.fc_block(self.conv_block(x).flatten(1))


class HSI_CNN3D(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(1, 32, (3,3,3)), nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32, 64, (3,3,3)), nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
        )
        self.fc_block = nn.Sequential(
            nn.Linear(64*13*5*5, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.fc_block(self.conv_block(x).flatten(1))


class HSI_ViT(nn.Module):
    def __init__(self, num_classes, image_size=15, patch_size=3, input_channels=30,
                 embed_dim=128, depth=4, heads=8, mlp_dim=256, dropout=0.1):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_size         = patch_size
        self.pos_embedding      = nn.Parameter(torch.randn(1, num_patches+1, embed_dim))
        self.patch_to_embedding = nn.Linear(input_channels * patch_size**2, embed_dim)
        self.cls_token          = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout            = nn.Dropout(dropout)
        enc = nn.TransformerEncoderLayer(embed_dim, heads, mlp_dim, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=depth)
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(self, img):
        B, C, H, W = img.shape
        p = self.patch_size
        x = img.unfold(2,p,p).unfold(3,p,p)
        x = x.contiguous().view(B,C,-1,p,p).permute(0,2,1,3,4).contiguous().view(B,-1,C*p*p)
        x = self.patch_to_embedding(x)
        cls = self.cls_token.expand(B,-1,-1)
        x   = torch.cat((cls,x),1) + self.pos_embedding
        return self.mlp_head(self.transformer(self.dropout(x))[:,0])


def _build_model(arch: str, num_classes: int) -> nn.Module:
    return {"cnn2d": HSI_CNN, "cnn3d": HSI_CNN3D, "vit": HSI_ViT}[arch](num_classes)

# ──────────────────────────────────────────────────────────────────────────────
# Section 4 — Preprocessing helpers
# ──────────────────────────────────────────────────────────────────────────────

def identify_dataset(filepath: str) -> "str | None":
    f = os.path.basename(filepath).lower()
    if "indian"  in f: return "indian_pines"
    if "pavia"   in f: return "pavia_university"
    if "salinas" in f: return "salinas"
    return None

def load_mat(filepath: str) -> np.ndarray:
    mat = scipy.io.loadmat(filepath)
    candidates = [(k,v) for k,v in mat.items()
                  if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim==3]
    if not candidates:
        raise ValueError("Dosyada 3-boyutlu dizi bulunamadı.")
    return max(candidates, key=lambda kv: kv[1].size)[1].astype(np.float32)

def preprocess(img: np.ndarray) -> np.ndarray:
    h,w,b = img.shape
    flat  = img.reshape(-1,b).astype(np.float32)
    flat  = (flat - flat.min()) / (flat.max() - flat.min() + 1e-8)   # normalize
    img_n = flat.reshape(h,w,b)
    img_g = scipy_gauss(img_n, sigma=(0.5,0.5,0), mode="reflect").astype(np.float32)
    return PCA(n_components=30).fit_transform(img_g.reshape(-1,b)).reshape(h,w,30).astype(np.float32)

def extract_patches(img: np.ndarray, gt: np.ndarray, window: int = 15):
    m  = window // 2
    p  = np.pad(img, ((m,m),(m,m),(0,0)), mode="reflect")
    rs, cs = np.where(gt > 0)
    X = np.stack([p[r:r+window, c:c+window, :] for r,c in zip(rs,cs)])
    return X, rs, cs

def _to_tensor(X: np.ndarray, arch: str) -> torch.Tensor:
    t = np.transpose(X,(0,3,1,2))
    if arch == "cnn3d": t = t[:,np.newaxis]
    return torch.tensor(t, dtype=torch.float32)

def _metrics(y_true, y_pred, nc):
    oa = accuracy_score(y_true, y_pred)
    k  = cohen_kappa_score(y_true, y_pred)
    aa = float(np.mean([accuracy_score(y_true[y_true==c], y_pred[y_true==c])
                        for c in range(nc) if (y_true==c).sum()>0]))
    return {
        "OA": oa, "AA": aa, "Kappa": k,
        "F1": f1_score(y_true,y_pred,average="macro",zero_division=0),
        "Precision": precision_score(y_true,y_pred,average="macro",zero_division=0),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Section 5 — PreprocessWorker
# ──────────────────────────────────────────────────────────────────────────────

class PreprocessWorker(QThread):
    done  = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, filepath, dataset_key):
        super().__init__()
        self.filepath = filepath; self.dataset_key = dataset_key

    def run(self):
        try:
            img     = load_mat(self.filepath)
            gt_path = os.path.join(PROCESSED_DIR, DATASETS[self.dataset_key]["gt_file"])
            if not os.path.exists(gt_path):
                raise FileNotFoundError(
                    f"GT dosyası bulunamadı:\n{gt_path}\n"
                    "datasets/processed/ klasörünü kontrol edin.")
            self.done.emit(preprocess(img), np.load(gt_path))
        except Exception as e:
            self.error.emit(str(e))

# ──────────────────────────────────────────────────────────────────────────────
# Section 6 — InferenceWorker
# ──────────────────────────────────────────────────────────────────────────────

class InferenceWorker(QThread):
    result_ready   = pyqtSignal(str, object, object, dict)
    error_occurred = pyqtSignal(str, str)

    def __init__(self, model_key, arch, pth_path, dataset_key, img_pca, gt):
        super().__init__()
        self.model_key=model_key; self.arch=arch; self.pth_path=pth_path
        self.dataset_key=dataset_key; self.img_pca=img_pca; self.gt=gt

    def run(self):
        try:
            nc    = DATASETS[self.dataset_key]["num_classes"]
            model = _build_model(self.arch, nc)
            try:   state = torch.load(self.pth_path, map_location="cpu", weights_only=True)
            except Exception: state = torch.load(self.pth_path, map_location="cpu")
            model.load_state_dict(state); model.eval()

            X, rows, cols = extract_patches(self.img_pca, self.gt)
            X_t = _to_tensor(X, self.arch)

            preds = np.concatenate([
                model(X_t[i:i+512]).argmax(1).numpy()
                for i in range(0, len(X_t), 512)
            ] if len(X_t) > 0 else [np.array([])])

            pred_map = np.zeros_like(self.gt, dtype=np.int32)
            pred_map[rows, cols] = preds.astype(int) + 1

            self.result_ready.emit(
                self.model_key, pred_map, self.gt.copy(),
                _metrics(self.gt[rows,cols].astype(int)-1, preds.astype(int), nc)
            )
        except Exception as e:
            self.error_occurred.emit(self.model_key, str(e))

# ──────────────────────────────────────────────────────────────────────────────
# Section 7 — MapPanel  (QFrame-based card)
# ──────────────────────────────────────────────────────────────────────────────

class MapPanel(QFrame):
    """White card displaying one model's classification result."""

    _IDLE    = "card"
    _RUNNING = "card_accent_blue"
    _DONE    = "card_accent_green"
    _ERROR   = "card_accent_red"

    def __init__(self, model_key: str, label: str, mode: str, parent=None):
        super().__init__(parent)
        self.model_key = model_key
        self.label     = label
        self.mode      = mode
        self.setMinimumWidth(280)
        self._set_state(self._IDLE)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # header row
        hdr = QHBoxLayout()
        title = QLabel(self.label)
        title.setFont(QFont("Arial", 11, QFont.Weight.Bold))

        badge = QLabel(self.mode)
        if self.mode == "supervised":
            bc, tc = "#e0e7ff", "#4338ca"
        elif self.mode == "semi-supervised":
            bc, tc = "#ffedd5", "#c2410c"
        else:  # federated
            bc, tc = "#d1fae5", "#065f46"
        badge.setStyleSheet(
            f"background:{bc};color:{tc};border-radius:5px;"
            f"padding:2px 8px;font-size:10px;font-weight:600;border:none;"
        )
        hdr.addWidget(title); hdr.addStretch(); hdr.addWidget(badge)
        layout.addLayout(hdr)

        # thin separator
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background:#f1f5f9;border:none;max-height:1px;")
        layout.addWidget(sep)

        # placeholder (shown while waiting)
        self.placeholder = QLabel("Bekleniyor…")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setMinimumHeight(190)
        self.placeholder.setStyleSheet(
            "background:#f8fafc;color:#94a3b8;border-radius:8px;"
            "font-size:14px;border:1px dashed #e2e8f0;"
        )
        layout.addWidget(self.placeholder)

        # matplotlib canvas (hidden until result arrives)
        self.fig    = Figure(figsize=(4.6, 2.7), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumHeight(190)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.hide()
        layout.addWidget(self.canvas)

        # metrics row
        self.metrics_lbl = QLabel("")
        self.metrics_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.metrics_lbl.setStyleSheet(
            "color:#374151;font-size:12px;font-weight:600;padding:2px;border:none;"
        )
        layout.addWidget(self.metrics_lbl)

    # ── state helpers ────────────────────────────────────────────────────────
    def _set_state(self, name: str):
        self.setObjectName(name)
        self.setStyleSheet(self.styleSheet())  # force re-polish
        self.style().unpolish(self)
        self.style().polish(self)

    # ── public API ───────────────────────────────────────────────────────────
    def show_waiting(self):
        self.placeholder.setText("Hesaplanıyor…")
        self.placeholder.setStyleSheet(
            "background:#eef2ff;color:#6366f1;border-radius:8px;"
            "font-size:14px;border:1px dashed #a5b4fc;"
        )
        self._set_state(self._RUNNING)

    def show_result(self, pred_map: np.ndarray, gt_map: np.ndarray, metrics: dict):
        nc   = int(gt_map.max())
        cmap = _make_cmap(nc)

        def _disp(arr):
            d = arr.astype(float) - 1
            d[gt_map == 0] = np.nan
            return d

        self.fig.clear()
        ax1 = self.fig.add_subplot(1,2,1)
        ax2 = self.fig.add_subplot(1,2,2)
        kw  = dict(cmap=cmap, vmin=0, vmax=nc-1, interpolation="nearest")
        ax1.imshow(_disp(gt_map),   **kw)
        ax2.imshow(_disp(pred_map), **kw)
        for ax, t in [(ax1,"Ground Truth"),(ax2,"Predicted")]:
            ax.set_title(t, fontsize=8, pad=3, color="#374151", fontweight="semibold")
            ax.axis("off")
        self.fig.patch.set_facecolor("white")
        self.canvas.draw()

        self.placeholder.hide()
        self.canvas.show()
        oa_pct = metrics["OA"] * 100
        aa_pct = metrics["AA"] * 100
        self.metrics_lbl.setText(
            f"OA: {oa_pct:.2f}%   AA: {aa_pct:.2f}%   κ: {metrics['Kappa']:.4f}"
        )
        self._set_state(self._DONE)

    def show_error(self, msg: str):
        short = msg[:110] + "…" if len(msg) > 110 else msg
        self.placeholder.setText(f"Hata:\n{short}")
        self.placeholder.setStyleSheet(
            "background:#fff5f5;color:#dc2626;border-radius:8px;"
            "font-size:11px;padding:8px;border:1px dashed #fca5a5;"
        )
        self._set_state(self._ERROR)

    def reset(self):
        self.canvas.hide()
        self.placeholder.show()
        self.placeholder.setText("Bekleniyor…")
        self.placeholder.setStyleSheet(
            "background:#f8fafc;color:#94a3b8;border-radius:8px;"
            "font-size:14px;border:1px dashed #e2e8f0;"
        )
        self.metrics_lbl.setText("")
        self._set_state(self._IDLE)

# ──────────────────────────────────────────────────────────────────────────────
# Section 8 — SidePanel
# ──────────────────────────────────────────────────────────────────────────────

class SidePanel(QFrame):
    classify_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setFixedWidth(240)
        self.filepath    = None
        self.dataset_key = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # title
        t = QLabel("HSI Sınıflandırma")
        t.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        t.setStyleSheet("color:#1e1b4b;border:none;")
        layout.addWidget(t)

        sub = QLabel("Multithreaded Karşılaştırmalı Analiz")
        sub.setStyleSheet("color:#7c3aed;font-size:11px;border:none;")
        layout.addWidget(sub)

        layout.addWidget(_hline())

        # file select
        self.select_btn = QPushButton("  Dosya Seç (.mat)")
        self.select_btn.setObjectName("ghost_btn")
        self.select_btn.clicked.connect(self._select_file)
        layout.addWidget(self.select_btn)

        self.file_lbl = QLabel("Dosya seçilmedi")
        self.file_lbl.setWordWrap(True)
        self.file_lbl.setStyleSheet("color:#64748b;font-size:11px;border:none;")
        layout.addWidget(self.file_lbl)

        self.dataset_lbl = QLabel("")
        self.dataset_lbl.setWordWrap(True)
        self.dataset_lbl.setStyleSheet(
            "color:#4338ca;font-weight:bold;font-size:12px;border:none;"
        )
        layout.addWidget(self.dataset_lbl)

        self.model_status_lbl = QLabel("")
        self.model_status_lbl.setWordWrap(True)
        self.model_status_lbl.setStyleSheet("color:#374151;font-size:11px;border:none;")
        layout.addWidget(self.model_status_lbl)

        layout.addWidget(_hline())

        self.classify_btn = QPushButton("Classify")
        self.classify_btn.setObjectName("primary_btn")
        self.classify_btn.setEnabled(False)
        self.classify_btn.clicked.connect(self.classify_requested.emit)
        layout.addWidget(self.classify_btn)

        # progress
        prog_lbl = QLabel("İlerleme")
        prog_lbl.setStyleSheet("color:#64748b;font-size:11px;font-weight:600;border:none;")
        layout.addWidget(prog_lbl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, _TOTAL_MODELS)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        self.progress_lbl = QLabel(f"0 / {_TOTAL_MODELS} model")
        self.progress_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.progress_lbl.setStyleSheet("color:#94a3b8;font-size:10px;border:none;")
        layout.addWidget(self.progress_lbl)

        self.status_lbl = QLabel("Hazır")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet(
            "color:#64748b;font-size:11px;font-style:italic;border:none;"
        )
        layout.addWidget(self.status_lbl)

        layout.addStretch()

        # footer version tag
        ver = QLabel("HSI Classifier v1.0")
        ver.setStyleSheet("color:#cbd5e1;font-size:10px;border:none;")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(ver)

    # ── file handling ────────────────────────────────────────────────────────
    def _select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "MAT Dosyası Seç", "", "MAT Files (*.mat)")
        if not path:
            return
        self.filepath = path
        self.file_lbl.setText(os.path.basename(path))

        key = identify_dataset(path)
        if key:
            self.dataset_key = key
            self.dataset_lbl.setText(f"Dataset: {DATASETS[key]['display']}")
            self.dataset_lbl.setStyleSheet(
                "color:#4338ca;font-weight:bold;font-size:12px;border:none;"
            )
            self._refresh_model_status(key)
        else:
            self.dataset_key = None
            self.dataset_lbl.setText("⚠ Dataset tanınamadı")
            self.dataset_lbl.setStyleSheet(
                "color:#dc2626;font-weight:bold;font-size:12px;border:none;"
            )
            self.model_status_lbl.setText("")
            self.classify_btn.setEnabled(False)

    def _refresh_model_status(self, key: str):
        prefix = DATASETS[key]["prefix"]
        found  = sum(
            os.path.exists(os.path.join(MODELS_DIR, f"{c['file_prefix']}_{prefix}.pth"))
            for c in MODEL_CONFIGS
        )
        total = _TOTAL_MODELS
        if   found == total: txt, color = f"✓  {total} / {total} model hazır", "#059669"
        elif found > 0:      txt, color = f"⚠  {found} / {total} model ({total-found} eksik)", "#d97706"
        else:                txt, color = "✗  Model bulunamadı",                "#dc2626"
        self.model_status_lbl.setText(txt)
        self.model_status_lbl.setStyleSheet(
            f"color:{color};font-size:11px;font-weight:bold;border:none;"
        )
        self.classify_btn.setEnabled(found > 0)

    # ── public API ──────────────────────────────────────────────────────────
    def set_running(self, running: bool):
        self.classify_btn.setEnabled(not running)
        self.select_btn.setEnabled(not running)
        if running:
            self.progress_bar.setValue(0)
            self.progress_lbl.setText(f"0 / {_TOTAL_MODELS} model")

    def update_progress(self, value: int):
        self.progress_bar.setValue(value)
        self.progress_lbl.setText(f"{value} / {_TOTAL_MODELS} model")

    def set_status(self, msg: str):
        self.status_lbl.setText(msg)

# ──────────────────────────────────────────────────────────────────────────────
# Section 9 — StatsPanel
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_LABELS = {
    "cnn2d_sup":  "CNN2D  —  Supervised",
    "cnn3d_sup":  "CNN3D  —  Supervised",
    "vit_sup":    "ViT      —  Supervised",
    "cnn2d_semi": "CNN2D  —  Semi-supervised",
    "cnn3d_semi": "CNN3D  —  Semi-supervised",
    "vit_semi":   "ViT      —  Semi-supervised",
    "cnn2d_fed":  "CNN2D  —  Federated",
    "cnn3d_fed":  "CNN3D  —  Federated",
    "vit_fed":    "ViT      —  Federated",
}

class StatsPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)

        # header
        hdr = QHBoxLayout()
        title = QLabel("Karşılaştırma Tablosu")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title.setStyleSheet("color:#1e1b4b;border:none;")
        self.toggle_btn = QPushButton("Gizle")
        self.toggle_btn.setObjectName("ghost_btn")
        self.toggle_btn.setFixedWidth(72)
        self.toggle_btn.clicked.connect(self._toggle)
        hdr.addWidget(title); hdr.addStretch(); hdr.addWidget(self.toggle_btn)
        layout.addLayout(hdr)

        sub = QLabel("Tüm modellerin OA, AA, Kappa, F1 ve Precision karşılaştırması  ·  "
                     "En yüksek OA → yeşil vurgu")
        sub.setStyleSheet("color:#64748b;font-size:11px;border:none;")
        layout.addWidget(sub)

        # table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Model", "OA (%)", "AA (%)", "Kappa", "F1", "Precision"]
        )
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 6):
            hh.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setMinimumHeight(100)
        self.table.setSizeAdjustPolicy(QTableWidget.SizeAdjustPolicy.AdjustToContents)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.table)
        self._visible = True

    def _toggle(self):
        self._visible = not self._visible
        self.table.setVisible(self._visible)
        self.toggle_btn.setText("Gizle" if self._visible else "Göster")

    def clear(self):
        self.table.setRowCount(0)

    def update(self, metrics_dict: dict):
        self.table.setRowCount(0)
        if not metrics_dict:
            return
        best_oa = max(m["OA"] for m in metrics_dict.values())
        order   = [
            "cnn2d_sup","cnn3d_sup","vit_sup",
            "cnn2d_semi","cnn3d_semi","vit_semi",
            "cnn2d_fed","cnn3d_fed","vit_fed",
        ]
        for key in order:
            if key not in metrics_dict:
                continue
            m   = metrics_dict[key]
            row = self.table.rowCount()
            self.table.insertRow(row)
            is_best = (m["OA"] == best_oa and len(metrics_dict) > 1)
            values = [
                _MODEL_LABELS[key],
                f"{m['OA']*100:.2f}",
                f"{m['AA']*100:.2f}",
                f"{m['Kappa']:.4f}",
                f"{m['F1']:.4f}",
                f"{m['Precision']:.4f}",
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if is_best:
                    item.setBackground(QColor("#d1fae5"))
                    item.setForeground(QColor("#065f46"))
                    font = item.font(); font.setBold(True); item.setFont(font)
                self.table.setItem(row, col, item)

        self.table.setVisible(True)
        self._visible = True
        self.toggle_btn.setText("Gizle")
        self._fit_table_height()

    def _fit_table_height(self):
        h = self.table.horizontalHeader().height() + 4
        for i in range(self.table.rowCount()):
            h += self.table.rowHeight(i)
        self.table.setFixedHeight(h)

# ──────────────────────────────────────────────────────────────────────────────
# Section 10 — MainWindow
# ──────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSI Classification — Comparative Analysis")
        self.resize(1460, 820)
        self._workers:     list = []
        self._prep_worker        = None
        self._completed:   int  = 0
        self._all_metrics: dict = {}
        self._init_ui()

    def _init_ui(self):
        # ── scroll wrapper ───────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.setCentralWidget(scroll)

        content = QWidget()
        content.setObjectName("scroll_content")
        scroll.setWidget(content)

        root = QVBoxLayout(content)
        root.setContentsMargins(14, 14, 14, 20)
        root.setSpacing(14)

        # ── header bar ───────────────────────────────────────────────────────
        header_frame = QFrame()
        header_frame.setFixedHeight(58)
        header_frame.setStyleSheet(
            "QFrame { background: qlineargradient("
            "  x1:0,y1:0,x2:1,y2:0,"
            "  stop:0 #6366f1, stop:1 #7c3aed"
            "); border-radius: 12px; }"
        )
        hdr_layout = QHBoxLayout(header_frame)
        hdr_layout.setContentsMargins(20, 0, 20, 0)

        app_title = QLabel("HSI Classification")
        app_title.setFont(QFont("Arial", 15, QFont.Weight.Bold))
        app_title.setStyleSheet("color:white;background:transparent;border:none;")

        app_sub = QLabel("Multithreaded Paralel Hiperspektral Görüntü Sınıflandırma")
        app_sub.setStyleSheet(
            "color:rgba(255,255,255,0.80);font-size:12px;background:transparent;border:none;"
        )
        hdr_layout.addWidget(app_title)
        hdr_layout.addSpacing(16)
        hdr_layout.addWidget(app_sub)
        hdr_layout.addStretch()
        root.addWidget(header_frame)

        # ── main row: side + result grid ─────────────────────────────────────
        main_row = QHBoxLayout()
        main_row.setSpacing(14)

        self.side = SidePanel()
        self.side.classify_requested.connect(self._start)
        main_row.addWidget(self.side, 0, Qt.AlignmentFlag.AlignTop)

        grid = QVBoxLayout()
        grid.setSpacing(10)
        self.map_panels: dict[str, MapPanel] = {}

        for row_label, row_color, keys in [
            ("SUPERVISED",      "#6366f1", ["cnn2d_sup",  "cnn3d_sup",  "vit_sup"]),
            ("SEMI-SUPERVISED", "#f97316", ["cnn2d_semi", "cnn3d_semi", "vit_semi"]),
            ("FEDERATED",       "#059669", ["cnn2d_fed",  "cnn3d_fed",  "vit_fed"]),
        ]:
            lbl = QLabel(row_label)
            lbl.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            lbl.setStyleSheet(
                f"color:{row_color};letter-spacing:2px;font-size:11px;border:none;"
            )
            grid.addWidget(lbl)

            row_h = QHBoxLayout()
            row_h.setSpacing(10)
            for key in keys:
                cfg   = next(c for c in MODEL_CONFIGS if c["key"] == key)
                panel = MapPanel(key, cfg["label"], cfg["mode"])
                self.map_panels[key] = panel
                row_h.addWidget(panel)
            grid.addLayout(row_h)

        main_row.addLayout(grid, 1)
        root.addLayout(main_row, 1)

        # ── stats panel ──────────────────────────────────────────────────────
        self.stats = StatsPanel()
        root.addWidget(self.stats, 0)

    # ── classification flow ─────────────────────────────────────────────────

    def _start(self):
        if not self.side.filepath or not self.side.dataset_key:
            return
        for p in self.map_panels.values(): p.reset()
        self._all_metrics.clear()
        self._completed = 0
        self._workers.clear()
        self.stats.clear()
        self.side.set_running(True)
        self.side.set_status("Preprocessing çalışıyor…")

        self._prep_worker = PreprocessWorker(self.side.filepath, self.side.dataset_key)
        self._prep_worker.done.connect(self._on_prep_done)
        self._prep_worker.error.connect(self._on_prep_error)
        self._prep_worker.start()

    def _on_prep_done(self, img_pca, gt):
        self.side.set_status("Modeller paralel çalışıyor…")
        dk     = self.side.dataset_key
        prefix = DATASETS[dk]["prefix"]

        for cfg in MODEL_CONFIGS:
            panel    = self.map_panels[cfg["key"]]
            pth_path = os.path.join(MODELS_DIR, f"{cfg['file_prefix']}_{prefix}.pth")
            if not os.path.exists(pth_path):
                panel.show_error(f"Dosya bulunamadı:\n{os.path.basename(pth_path)}")
                self._tick(); continue
            panel.show_waiting()
            w = InferenceWorker(cfg["key"], cfg["arch"], pth_path, dk, img_pca, gt)
            w.result_ready.connect(self._on_result)
            w.error_occurred.connect(self._on_error)
            self._workers.append(w)
            w.start()

    def _on_prep_error(self, msg: str):
        self.side.set_status(f"Preprocessing hatası:\n{msg}")
        self.side.set_running(False)

    def _on_result(self, model_key, pred_map, gt_map, metrics):
        self.map_panels[model_key].show_result(pred_map, gt_map, metrics)
        self._all_metrics[model_key] = metrics
        self._tick()

    def _on_error(self, model_key, msg):
        self.map_panels[model_key].show_error(msg)
        self._tick()

    def _tick(self):
        self._completed += 1
        self.side.update_progress(self._completed)
        if self._completed >= _TOTAL_MODELS:
            self.side.set_running(False)
            self.side.set_status("Tamamlandı ✓")
            self.stats.update(self._all_metrics)

# ──────────────────────────────────────────────────────────────────────────────
# Section 11 — Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("background:#f1f5f9;border:none;max-height:1px;")
    return f

# ──────────────────────────────────────────────────────────────────────────────
# Section 12 — Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)
    app.setFont(QFont("Arial", 11))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
