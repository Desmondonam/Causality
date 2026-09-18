"""Central paths and constants shared by every stage of the pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "data.csv"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

for _dir in (MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

TARGET = "diagnosis"
ID_COLUMN = "id"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columns used for the causal-graph stage (a manageable, domain-meaningful
# subset of the 30 features, mirroring the "mean" vs "worst" structure of
# the original UCI schema).
CAUSAL_FEATURES = [
    "radius_mean",
    "perimeter_mean",
    "area_mean",
    "concavity_mean",
    "concave points_mean",
    "radius_worst",
    "perimeter_worst",
    "area_worst",
    "concavity_worst",
    "concave points_worst",
]

N_TOP_FEATURES = 15
