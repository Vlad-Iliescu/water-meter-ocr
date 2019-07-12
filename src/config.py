import os
from configparser import ConfigParser

config = ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '..', 'config.ini'))
config.default_section = 'MAIN'
