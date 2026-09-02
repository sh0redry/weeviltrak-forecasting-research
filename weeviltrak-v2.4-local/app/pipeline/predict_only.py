"""
Original AWS-backed example (requires the private production dependencies):
python -m app.pipeline.predict_only --config-path app/config/weeviltrak_v2.4.yml --test-date 2027-03-02

For the credential-free local workflow use ``python -m app.local.cli predict``.
"""
import argparse
import pandas as pd

from app.pipeline.train_predict import WeevilTrakPipeline


def main():
    parser = argparse.ArgumentParser(description="Load model and run WeevilTrak prediction for a given test date.)")
    parser.add_argument("--config-path", required=True, help="Path to YAML config, e.g. configs/weeviltrak.yaml")
    parser.add_argument("--test-date", required=True, help="Test date (YYYY-MM-DD or any pandas-parseable date)")
    args = parser.parse_args()

    test_date_str = pd.to_datetime(args.test_date).strftime("%Y-%m-%d")

    # 1) Init pipeline
    pipeline = WeevilTrakPipeline(config_path=args.config_path)

    # 2) Explicitly load model (as requested)
    model = pipeline.model_manager.load_model()
    print(f"Model loaded: {type(model)}")

    # 3) Run prediction (pipeline handles weather pull + features + S3 write, etc.)
    pred_df = pipeline.run_prediction(test_date=test_date_str)

if __name__ == "__main__":
    main()
