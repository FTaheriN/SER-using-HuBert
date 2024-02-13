import yaml

def read_yaml_config():
    # Load config file
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config