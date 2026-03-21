import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

# BC Catalogue
BC_CATALOGUE_BASE_URL = os.getenv(
    "BC_CATALOGUE_BASE_URL",
    "https://catalogue.data.gov.bc.ca/api/3/action"
)

# Paths
DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
PROCESSED_DIR = f"{DATA_DIR}/processed"
LABELS_DIR = f"{DATA_DIR}/labels"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

# Classifier settings
LABELLED_DATA_PATH = f"{LABELS_DIR}/bc_datasets_labelled.csv"
MODEL_SAVE_PATH = f"{MODELS_DIR}/rf_classifier.pkl"

# Search settings
SEARCH_TERMS = [
    "soil",
    "hydrology",
    "drainage",
    "moisture",
    "terrain elevation lidar",
    "tree species vegetation",
    "precipitation climate",
    "watershed",
    "land cover",
    "cadastral survey",
    "geology",
    "fauna wildlife species"
]

# Geographic focus area - Ucluelet BC bounding box
# Format: [min_longitude, min_latitude, max_longitude, max_latitude]
UCLUELET_BBOX = [-125.8, 48.8, -125.4, 49.0]

# Chanterelle relevant data types for classifier priority
CHANTERELLE_RELEVANT_TAGS = [
    "soil",
    "moisture",
    "hydrology",
    "forest",
    "vegetation",
    "precipitation",
    "terrain",
    "drainage",
    "species",
    "habitat"
]