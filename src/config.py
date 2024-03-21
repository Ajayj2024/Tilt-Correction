import yaml

def params():
    with open('src/config.yaml','r') as f:
        data = yaml.safe_load(f)
        
    return data

param = params
print(param)