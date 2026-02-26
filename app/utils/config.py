import yaml
import os

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../../config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print("Error loading config.yaml:", e)
        config = {}
    return config
