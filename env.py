from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATA_DOWNLOADED = Path.home() / "Downloads" / "the-circor-digiscope-phonocardiogram-dataset-1.0.3" / "training_data"
RAW_DATA_DIR = PROJECT_DIR / "data_raw.pickle"
DATA_FOR_ML = PROJECT_DIR / "downsampled_4x_data_for_ml"


FIGURES_DIR = PROJECT_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

RATE = 4000
DOWNSAMPLE_FACTOR = 4
RATE_DS = RATE // DOWNSAMPLE_FACTOR  # 1000
TYPES = ['TV', 'MV', 'PV', 'Phc', 'AV']
CLASSES = [0, 1, 2, 3, 4]
CATEGORY_NAMES = ['background', 'S1', 'systolic', 'S2', 'diastole']

NUMERIC_CATEGORIZED_MURMUR = {
    'Present' : 0,
    'Absent' : 1,
    'Unknown' : 2,
}
