import  yaml

def check():
    print("Check is correct")

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

        return config