'''
This module contains the configuration for the weevilltrak model.
'''
import yaml
import sys
sys.path.append("..")



class WeevillTrakConfig:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.features = self.config['features']
        self.target = self.config['target']
        self.model_params = self.config['model_parameters']


    def load_config(self, path) -> dict:
        """
        Loads a configuration file.

        Parameters:
            path (str): The file path to the YAML configuration file.

        Returns:
            dict: A dictionary containing the configuration.
        """
        with open(path, "r", encoding="UTF-8") as stream:
            return yaml.safe_load(stream)

if __name__ == "__main__":
    config = WeevillTrakConfig("weeviltrak_v2.3.yml")
    print(config)
    print("-----------------")
    print(config.model_params)
