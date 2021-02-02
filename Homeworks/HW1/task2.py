#!/usr/bin/env python3
import pandas as pd
import numpy as np

csv_thing = """rating,easy,ai,systems,theory,morning
 2,True,True,False,True,False
 2,True,True,False,True,False
 2,False,True,False,False,False
 2,False,False,False,True,False
 2,False,True,True,False,True
 1,True,True,False,False,False
 1,True,True,False,True,False
 1,False,True,False,True,False
 0,False,False,False,False,True
 0,True,False,False,True,True
 0,False,True,False,True,False
 0,True,True,True,True,True
-1,True,True,True,False,True
-1,False,False,True,True,False
-1,False,False,True,False,True
-1,True,False,True,False,True
-2,False,False,True,True,False
-2,False,True,True,False,True
-2,True,False,True,False,False
-2,True,False,True,False,True
"""
file = open("csv_thing.csv", 'w')
file.write(csv_thing)
file.close()
res = pd.read_csv("csv_thing.csv")
res['ok'] = np.where(res['rating'] >= 0, True, False)
print(res)
res.to_csv("task2_result", index=False)
