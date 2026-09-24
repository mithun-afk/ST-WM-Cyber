import sys

def main():
    try:
        from src2.config import get_config
        get_config()
        print("Config OK")

        from src2.data.schema import CANONICAL_FEATURES, MODEL_FEATURES, DataQualityGate
        assert len(CANONICAL_FEATURES) > 0
        assert len(MODEL_FEATURES) > 0
        gate = DataQualityGate()
        print("Schema OK")

        from src2.data.csv_loader import load_and_normalize_csv
        from src2.data.feature_engineering import prepare_model_features, engineer_features
        print("Data OK")

        from src2.models.world_model import STGWMModel
        model = STGWMModel()
        print("Model Object OK")

        from src2.models.inference import run_inference
        from src2.live.live_pipeline import LivePipeline
        print("Inference/Live OK")
        
        from src2.intelligence.mitre import match_indicators
        print("MITRE Intelligence OK")
        
        print("ALL VALIDATION PASSED!")
    except Exception as e:
        print(f"VALIDATION FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
