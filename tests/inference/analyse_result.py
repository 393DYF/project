import latex
import pandas as pd
from os import path
from warnings import filterwarnings
filterwarnings('ignore')
import sys
sys.path.append('../')
from utils.analyse_results import *

dirname = 'C:/Users/lenovo/Desktop/linkage/synthetic_data_release/tests/inference/tests'
cwd = path.dirname(__file__)
dpath = cwd
inference_gain = load_results_inference(dirname, dpath)
print(inference_gain)
test = pd.DataFrame(data=inference_gain)
# print(test)
test.to_csv("C:/Users/lenovo/Desktop/linkage/synthetic_data_release/inference_gain_try3_new.csv", mode='a+', index=None, header=None)
#models = ['SanitiserNHSk10', 'BayesianNet', 'PrivBayesEps1.0']
#fig = plt_per_target_pg(inference_gain, models, resFilter=('SensitiveAttribute', 'RACE'))