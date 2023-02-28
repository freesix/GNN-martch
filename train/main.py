import yaml
from config import get_config, print_unage


if __name__ == '__main__':

    config, unparsed = get_config()
    with open(config.config_path) as f:
        model_config = yaml.safe_load(f)
        print(model_config)
    
    if len(unparsed) > 0:
        print_unage()
        exit(1)
