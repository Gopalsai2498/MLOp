# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 19:54:29 2021

@author: anil.ms
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'sepal_lenth':2, 'sepal_width':9, 'petal_length':6, 'petal_width':5})

print(r.json())