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
FETCH_OUTPUT_DIR = f"{RAW_DIR}/fetched"

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
UCLUELET_BBOX = [-125.9, 48.8, -125.4, 49.0]

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

# Fetching settings
MAX_WFS_FEATURES = 5000     # cap per layer to avoid huge downloads
REQUEST_TIMEOUT = 30        # seconds before giving up on a request
RETRY_ATTEMPTS = 3          # how many times to retry a failed request

# File size limits
MAX_DOWNLOAD_MB = 200       # skip files larger than this

# WFS output format preferences in order of preference
WFS_FORMAT_PREFERENCE = [
    "application/json",
    "json",
    "geojson",
    "GML3"
]

# Keywords for optional BC OpenMaps WFS layer discovery.
# These are merged with classifier-selected catalogue datasets
# during fetch in pipeline.ingest.
WFS_DISCOVERY_KEYWORDS = [
    "water", "watershed", "aquifer",
    "road", "forest", "soil",
    "wildlife", "fish", "marine",
    "shoreline", "coastal", "riparian",
    "zoning", "administrative", "park",
    "terrain", "slope", "elevation"
]

# ChromaDB vector database settings
CHROMA_DB_DIR = "data/chromadb"
CHROMA_COLLECTION_NAME = "ucluelet_geodata"

# Text chunking settings
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 200      # overlap between chunks
MAX_CHUNKS_PER_DOC = 500 # safety cap for very large documents