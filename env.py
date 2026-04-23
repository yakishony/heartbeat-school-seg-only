from pathlib import Path

PROJECT_DIR = Path(__file__).parent
# Path to the downloaded Kaggle dataset directory
DATA_DOWNLOADED = Path.home() / "Downloads" / "the-circor-digiscope-phonocardiogram-dataset-1.0.3" / "training_data"
# Pickle cache of raw dataset (for fast reloading)
DATA_RAW_DIR = PROJECT_DIR / "data_raw.pickle"

DATA_FOR_ML_X2 = PROJECT_DIR / "downsampled_2x_data_for_ml"
DATA_FOR_ML_X4 = PROJECT_DIR / "downsampled_4x_data_for_ml"
# Active data path — processed data fed to the model
DATA_FOR_ML = DATA_FOR_ML_X4

FIGURES_DIR = PROJECT_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

# Original sampling rate and downsampling factor
RATE = 4000
DOWNSAMPLE_FACTOR = 4
RATE_DS = RATE // DOWNSAMPLE_FACTOR  # 1000

# Recording types (stethoscope auscultation locations)
TYPES = ['TV', 'MV', 'PV', 'Phc', 'AV']
# Segmentation categories
CLASSES = [0, 1, 2, 3, 4]
CATEGORY_NAMES = ['unannotated', 'S1', 'systolic', 'S2', 'diastole']

