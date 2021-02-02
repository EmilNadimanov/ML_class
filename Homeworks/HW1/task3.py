#!/usr/bin/env python3
import pandas as pd
from task1 import Tree 
data = pd.read_csv('result')
print(data)

def single_feature_score(data, goal, feature):
    truefalse = (data[goal]==data[feature])
    matches = truefalse.sum()
    total = truefalse.shape[0]
    return matches/total

for f in data.drop(columns=["rating"]).columns:
    print(f, single_feature_score(data, "ok", f))
