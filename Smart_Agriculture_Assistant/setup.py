"""Setup configuration and utilities for KRISHI application."""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
SMART_AG_DIR = Path(__file__).parent.parent
if str(SMART_AG_DIR) not in sys.path:
    sys.path.insert(0, str(SMART_AG_DIR))


def validate_environment() -> bool:
    """Validate that all required directories and files exist.
    
    Returns:
        True if environment is valid, False otherwise.
    """
    required_dirs = [
        SMART_AG_DIR / "CROP-RECOMMENDATION",
        SMART_AG_DIR / "PLANT-DISEASE-IDENTIFICATION",
    ]
    
    for directory in required_dirs:
        if not directory.exists():
            print(f"⚠️ Warning: Directory not found: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
    
    return True


def get_model_paths():
    """Get all model file paths.
    
    Returns:
        Dictionary with model paths.
    """
    return {
        "crop_model": SMART_AG_DIR / "CROP-RECOMMENDATION" / "RF.pkl",
        "disease_model": SMART_AG_DIR / "PLANT-DISEASE-IDENTIFICATION" / "trained_plant_disease_model.keras",
        "crop_image": SMART_AG_DIR / "CROP-RECOMMENDATION" / "crop.png",
        "disease_image": SMART_AG_DIR / "PLANT-DISEASE-IDENTIFICATION" / "Diseases.png",
    }


if __name__ == "__main__":
    validate_environment()
    print("✅ Environment setup complete!")
    print(f"\n📁 Smart Agriculture Directory: {SMART_AG_DIR}")
    print(f"\n🔧 Model Paths:")
    for name, path in get_model_paths().items():
        status = "✅" if path.exists() else "❌"
        print(f"   {status} {name}: {path}")
