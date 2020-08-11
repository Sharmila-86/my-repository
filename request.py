# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 20:29:56 2020

@author: sharm
"""

import requests
url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'radius_mean':1.5, 'area_mean':1.8, 'smoothness_mean':2.1})

print(r.json())