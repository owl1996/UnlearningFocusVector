
import yaml
import sys

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

sys.path.append(config["sys_path"])

from .MIA import MIA
from .SVC_MIA import SVC_MIA
