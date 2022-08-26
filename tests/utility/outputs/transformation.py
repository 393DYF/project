#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import csv
import json
import pandas as pd
#---------用pandas库，filename写你自己的--------------
filename= "/utility/outputs/utilityresult.json"
with open(filename,'r',encoding='utf-8') as f:
  trump_list=json.load(f)
frame=pd.DataFrame(trump_list)
print(frame)
frame.to_csv("C:/Users/lenovo/Desktop/linkage/synthetic_data_release/tests/utility/outputs/result.csv")