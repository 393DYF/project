import latex
import pandas as pd
from warnings import filterwarnings
filterwarnings('ignore')
import sys
sys.path.append('../../')
from utils.analyse_results import *
dirname = '../tests/linkage/tests/'
linkage_gain = load_results_linkage(dirname)
models = ['IndependentHistogram','BayesianNet','PrivBayes']
print(linkage_gain)
test = pd.DataFrame(data=linkage_gain)
# print(test)
test.to_csv("C:/Users/lenovo/Desktop/linkage/synthetic_data_release/linkage_gain_new_try3_new.csv", mode='a+', index=None, header=None)
#models = ['SanitiserNHSk10', 'BayesianNet', 'PrivBayesEps1.0']
#fig = plt_per_target_pg(linkage_gain, models, resFilter=('FeatureSet', 'Naive'))