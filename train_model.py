import argparse
import subprocess
import os
import numpy as np
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.model_selection import StratifiedKFold

params_list = [
    {'s': 0, 't': 0, 'c': 1},
    {'s': 0,  't': 2,  'c': 1,  'g': 0.1},
    {'s': 0,  't': 2,  'c': 10,    'g': 0.1},
    {'s': 0,  't': 2,  'c': 100,   'g': 0.01},
    {'s': 0,  't': 2,  'c': 10,  'g': 1}
]
mode = 'cv'

for i, params in enumerate(params_list):
    cmd = [
        "svm-train",
        "-q",
            "-s", str(params["s"]),
            "-t", str(params["t"]),
            "-c", str(params["c"])
        ]

    # 只有非 linear 才加 gamma
    if params["t"] != 0:
        cmd += ["-g", str(params["g"])]

    if mode == 'holdout':
        subprocess.run(cmd + [
            f".\\data\\holdout\\train.scale",
            f".\\models\\holdout\\model{i+1}.txt"
        ])
        
    elif mode == 'cv':
        for j in range(1, 6):
            subprocess.run(cmd + [
                f".\\data\\cross_validation\\train_fold{j}.scale",
                f".\\models\\cross_validation\\model{i+1}\\model{i+1}_fold{j}.txt"
            ])

    print(f"model{i+1} train done")

print("train model done")
