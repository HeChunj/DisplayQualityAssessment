import yaml


class ConfigManager:
    def __init__(self, config_file=None):
        self.config_file = config_file
        self.config = None

        if config_file:
            self.load_config(config_file)

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def get(self, section):
        return self.config.get(section)

    def set(self, section, key, value):
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value

    def save(self, config_file=None):
        if not config_file:
            config_file = self.config_file
        with open(config_file, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)
