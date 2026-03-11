from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data_raw"
DATA_FOR_ML = PROJECT_DIR / "data_for_ml"

FIGURES_DIR = PROJECT_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

RATE = 4000
TYPES = ['TV', 'MV', 'PV', 'Phc', 'AV']
