import argparse
import yaml
from evaluate_model import EvaluationConfig, ModelEvaluator


def load_config(config_path: str) -> EvaluationConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return EvaluationConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    parser.add_argument('config', type=str, help='Path to configuration YAML file')
    args = parser.parse_args()

    config = load_config(args.config)
    evaluator = ModelEvaluator(config)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
