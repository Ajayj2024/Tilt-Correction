import yaml

def parameters():
    with open('src/config.yaml','r') as f:
        data = yaml.safe_load(f)
        
    return data
