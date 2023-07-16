## Starting fresh
## File created by Karim

import os, sys, argparse
import configs.settings as settings
import torch
from utils.ShapeNet import ShapeNetDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

if __name__=="__main__":
    settings.init()
    
    ShapenetDS = ShapeNetDataset()
    
