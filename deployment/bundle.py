"""Memuat artefak model dan metadata hasil ekspor notebook."""
import json
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

def load_bundle():
    model_path = MODEL_DIR / "model_artifact.joblib"
    metadata_path = MODEL_DIR / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Artefak model belum tersedia. Jalankan cell ekspor pada DEPLOYMENT.md "
            "dari notebook setelah best_pipeline dan best_thresh tersedia."
        )
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata
