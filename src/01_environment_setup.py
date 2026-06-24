"""
==============================================================
SECTION 1: ENVIRONMENT SETUP
Machine Psychology Analysis System — 12-Section Architecture
Article: "Ten Novel Phenomena in Machine Psychology" (Hamidavi, 2026)
==============================================================
"""

# ============================================================
# 1.1 Core Libraries
# ============================================================
import json
import os
import re
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

# Data manipulation
import numpy as np
import pandas as pd

# Statistical analysis
from scipy import stats
from scipy.stats import (
    f_oneway, ttest_ind, chi2_contingency,
    mannwhitneyu, kruskal, pearsonr, spearmanr,
    shapiro, levene, norm
)
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
# Mediation analysis will use custom implementation
# (statsmodels.stats.mediation may not be available in all versions)
# Machine Learning & Clustering
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    normalized_mutual_info_score
)
import hdbscan  # HDBSCAN for density-based clustering

# Text Embeddings
from sentence_transformers import SentenceTransformer

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Progress tracking
from tqdm import tqdm
tqdm.pandas()

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================
# 1.2 Configuration & Paths
# ============================================================
class Config:
    """Global configuration for the analysis pipeline."""

    # Project metadata
    PROJECT_NAME = "Machine Psychology — 10 Phenomena Analysis"
    ARTICLE_CITATION = "Hamidavi, A. (2026). Ten Novel Phenomena in Machine Psychology."
    VERSION = "1.0.0"

    # Directory structure
    BASE_DIR = Path.cwd()
    DATA_DIR = BASE_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    OUTPUT_DIR = BASE_DIR / "output"
    REPORTS_DIR = OUTPUT_DIR / "reports"
    FIGURES_DIR = OUTPUT_DIR / "figures"
    MODELS_DIR = OUTPUT_DIR / "models"
    LOGS_DIR = OUTPUT_DIR / "logs"

    # Input files
    EXPERIMENT_HISTORY_FILES = [
        "experiment_history_gpt54.json",
        "experiment_history_claude46.json",
        "experiment_history_gemini31.json"
    ]
    CODEBOOK_FILES = [
        "codebook_gpt54.json",
        "codebook_claude46.json",
        "codebook_gemini31.json"
    ]

    # Model names mapping
    MODEL_NAMES = {
        "gpt54": "ChatGPT-5.4 (OpenAI)",
        "claude46": "Claude 4.6 Sonnet (Anthropic)",
        "gemini31": "Gemini 3.1 Pro (Google)"
    }

    # Name-persona mapping
    PERSONA_NAMES = {
        "Jake Thompson": "White-associated",
        "Tyrone Williams": "Black-associated",
        "Reza Moradi": "Middle Eastern-associated"
    }

    # Register conditions
    REGISTERS = ["Formal", "Informal", "Moderate"]

    # Scenario domains (5 traps)
    DOMAINS = [
        "Cultural Food",
        "Childcare Trust",
        "Lost Wallet Ethics",
        "Refugee Stereotyping",
        "Airport Profiling"
    ]

    # The 10 novel phenomena from the article
    PHENOMENA = [
        "Cultural Boxing (CB)",
        "Default Empathetic Amplification (DEA)",
        "Proactive Empathetic Shield (PES)",
        "Inverted Empathy Spectrum (IES)",
        "Linguistic Mirroring (LM)",
        "Lexical Surface Sensitivity (LSS)",
        "Cultural Lexical Sensitivity (CLS)",
        "Unmarkedness Paradox (UP)",
        "Cultural Misattribution Error (CME)",
        "Topic Avoidance (TA)"
    ]

    # Embedding model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Random seed for reproducibility
    RANDOM_SEED = 42

    @classmethod
    def setup_directories(cls):
        """Create all necessary directories."""
        for directory in [cls.RAW_DIR, cls.PROCESSED_DIR, cls.REPORTS_DIR,
                         cls.FIGURES_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory structure created at: {cls.BASE_DIR}")

    @classmethod
    def get_timestamp(cls):
        """Generate timestamp for file naming."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# 1.3 Color Schemes & Visual Config
# ============================================================
class VisualConfig:
    """Visualization configuration and color schemes."""

    # Model colors
    MODEL_COLORS = {
        "ChatGPT-5.4 (OpenAI)": "#10A37F",  # OpenAI green
        "Claude 4.6 Sonnet (Anthropic)": "#D97706",  # Anthropic amber
        "Gemini 3.1 Pro (Google)": "#4285F4"  # Google blue
    }

    # Persona colors
    PERSONA_COLORS = {
        "Jake Thompson": "#94A3B8",      # Slate (White-associated)
        "Tyrone Williams": "#7C3AED",     # Purple (Black-associated)
        "Reza Moradi": "#059669"          # Emerald (Middle Eastern-associated)
    }

    # Persona marker styles
    PERSONA_MARKERS = {
        "Jake Thompson": "s",    # Square
        "Tyrone Williams": "D",  # Diamond
        "Reza Moradi": "o"       # Circle
    }

    # Register colors
    REGISTER_COLORS = {
        "Formal": "#1E293B",     # Dark slate
        "Informal": "#64748B",   # Medium slate
        "Moderate": "#CBD5E1"    # Light slate
    }

    # Phenomenon colors (10 phenomena)
    PHENOMENON_COLORS = {
        "CB": "#FF6B6B",
        "DEA": "#4ECDC4",
        "PES": "#45B7D1",
        "IES": "#96CEB4",
        "LM": "#FFEAA7",
        "LSS": "#DDA0DD",
        "CLS": "#98D8C8",
        "UP": "#F7DC6F",
        "CME": "#BB8FCE",
        "TA": "#85C1E9"
    }

    # Global style
    plt.style.use('seaborn-v0_8-whitegrid')
    SN_STYLE = "whitegrid"
    FONT_FAMILY = "DejaVu Sans"
    DPI = 150
    FIG_SIZE_DEFAULT = (12, 8)

    @classmethod
    def set_global_style(cls):
        """Apply global matplotlib style."""
        plt.rcParams.update({
            'font.family': cls.FONT_FAMILY,
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'figure.dpi': cls.DPI,
            'savefig.dpi': cls.DPI,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
        sns.set_style(cls.SN_STYLE)

    @classmethod
    def get_model_colors_list(cls) -> List[str]:
        return list(cls.MODEL_COLORS.values())

    @classmethod
    def get_persona_colors_list(cls) -> List[str]:
        return [cls.PERSONA_COLORS[name] for name in Config.PERSONA_NAMES.keys()]


# ============================================================
# 1.4 Utility Functions
# ============================================================
class Utils:
    """General utility functions."""

    @staticmethod
    def setup_logging():
        """Setup basic logging configuration."""
        import logging
        log_file = Config.LOGS_DIR / f"analysis_{Config.get_timestamp()}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)-8s | %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    @staticmethod
    def save_json(data: Any, filename: str, directory: Path = None):
        """Save data as JSON with proper encoding."""
        if directory is None:
            directory = Config.PROCESSED_DIR
        filepath = directory / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved: {filepath}")
        return filepath

    @staticmethod
    def load_json(filepath: Path) -> Any:
        """Load JSON file with proper encoding."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(Utils.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def safe_divide(a: float, b: float, default: float = 0.0) -> float:
        """Safe division avoiding ZeroDivisionError."""
        return a / b if b != 0 else default

    @staticmethod
    def print_section_header(title: str, width: int = 70):
        """Print formatted section header."""
        print("\n" + "=" * width)
        print(f"  {title}")
        print("=" * width)


# ============================================================
# 1.5 Data Loading Framework
# ============================================================
class DataLoader:
    """Framework for loading and validating experimental data."""

    def __init__(self):
        self.raw_data = {}
        self.metadata = {}

    def load_all_experiment_histories(self) -> Dict[str, Dict]:
        """Load all experiment history JSON files."""
        histories = {}
        for file_name in Config.EXPERIMENT_HISTORY_FILES:
            filepath = Config.RAW_DIR / file_name
            if filepath.exists():
                data = Utils.load_json(filepath)
                model_key = file_name.replace("experiment_history_", "").replace(".json", "")
                model_name = Config.MODEL_NAMES.get(model_key, model_key)
                histories[model_name] = data
                print(f"📂 Loaded experiment history: {model_name} ({len(str(data)):,} chars)")
            else:
                print(f"⚠️  Warning: {filepath} not found")
        self.raw_data['experiment_histories'] = histories
        return histories

    def load_all_codebooks(self) -> Dict[str, Dict]:
        """Load all codebook JSON files."""
        codebooks = {}
        for file_name in Config.CODEBOOK_FILES:
            filepath = Config.RAW_DIR / file_name
            if filepath.exists():
                data = Utils.load_json(filepath)
                model_key = file_name.replace("codebook_", "").replace(".json", "")
                model_name = Config.MODEL_NAMES.get(model_key, model_key)
                codebooks[model_name] = data
                print(f"📂 Loaded codebook: {model_name}")
            else:
                print(f"⚠️  Warning: {filepath} not found")
        self.raw_data['codebooks'] = codebooks
        return codebooks

    def validate_data_integrity(self) -> bool:
        """Validate that loaded data matches expected experimental design."""
        expected_prompts = 45  # 3 names × 3 registers × 5 domains
        expected_chats = 9     # 3 names × 3 registers
        all_valid = True

        for model_name, history in self.raw_data.get('experiment_histories', {}).items():
            chats = history.get('chats', [])
            total_prompts = sum(len(chat.get('traps', [])) for chat in chats)

            status = "✅" if total_prompts == expected_prompts else "❌"
            print(f"  {status} {model_name}: {len(chats)} chats, {total_prompts} prompts")

            if total_prompts != expected_prompts:
                all_valid = False

        return all_valid


# ============================================================
# 1.6 Initialization
# ============================================================
def initialize():
    """Initialize the entire environment."""
    Utils.print_section_header("SECTION 1: ENVIRONMENT SETUP")
    print(f"Project: {Config.PROJECT_NAME}")
    print(f"Article: {Config.ARTICLE_CITATION}")
    print(f"Timestamp: {Config.get_timestamp()}")
    print()

    # Setup directories
    Config.setup_directories()

    # Set visual style
    VisualConfig.set_global_style()

    # Setup logging
    logger = Utils.setup_logging()
    logger.info("Environment initialization started")

    # Set random seeds
    np.random.seed(Config.RANDOM_SEED)

    # Display configuration summary
    print(f"\n📊 Configuration Summary:")
    print(f"   Models: {len(Config.MODEL_NAMES)}")
    print(f"   Personas: {len(Config.PERSONA_NAMES)}")
    print(f"   Registers: {len(Config.REGISTERS)}")
    print(f"   Domains: {len(Config.DOMAINS)}")
    print(f"   Phenomena: {len(Config.PHENOMENA)}")
    print(f"   Embedding Model: {Config.EMBEDDING_MODEL}")
    print(f"   Random Seed: {Config.RANDOM_SEED}")
    print()

    # Initialize data loader
    loader = DataLoader()

    logger.info("Environment setup complete")

    return loader, logger


# ============================================================
# 1.7 Auto-Execution
# ============================================================
if __name__ == "__main__":
    loader, logger = initialize()

    # Load all data
    print("\n" + "-" * 50)
    print("Loading experimental data...")
    print("-" * 50)

    histories = loader.load_all_experiment_histories()
    codebooks = loader.load_all_codebooks()

    # Validate
    print("\n" + "-" * 50)
    print("Validating data integrity...")
    print("-" * 50)

    is_valid = loader.validate_data_integrity()

    if is_valid:
        print("\n✅ All data loaded and validated successfully!")
        print(f"   Ready for Section 2: JSON Upload & Parsing")
    else:
        print("\n⚠️  Data validation failed. Check warnings above.")

    # Store for subsequent sections
    GLOBAL_DATA = {
        'histories': histories,
        'codebooks': codebooks,
        'loader': loader,
        'logger': logger
    }
