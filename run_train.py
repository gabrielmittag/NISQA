# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:52:15 2020

@author: Gabriel
"""

import yaml
from nisqa.NISQA_model import nisqaModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--yaml', required=True, type=str, help='YAML file with config')

args = parser.parse_args()
args = vars(args)

if __name__ == "__main__":
    
    with open(args['yaml'], "r") as ymlfile:
        args_yaml = yaml.load(ymlfile, Loader=yaml.FullLoader)
    args = {**args_yaml, **args}
    
    nisqa = nisqaModel(args)
    nisqa.train()
































