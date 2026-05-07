import argparse
import subprocess
import os
import numpy as np
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

def calc_metrics(y_true, y_pred):
    # Labels are 1 (win) and -1 (lose)
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
    TP = cm[0, 0] # Actual 1, Pred 1
    FN = cm[0, 1] # Actual 1, Pred -1
    FP = cm[1, 0] # Actual -1, Pred 1
    TN = cm[1, 1] # Actual -1, Pred -1
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0 # TPR
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    tnr = TN / (FP + TN) if (FP + TN) > 0 else 0
    fnr = FN / (TP + FN) if (TP + FN) > 0 else 0
    
    return {
        'CM': cm,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall/TPR': recall,
        'FPR': fpr,
        'TNR': tnr,
        'FNR': fnr
    }

def print_metrics(m, title="Results", output_file=None):
    report = []
    report.append(f"\n--- {title} ---")
    report.append(f"Confusion Matrix:\n{m['CM']}")
    report.append(f"Accuracy:  {m['Accuracy']:.4f}")
    report.append(f"Precision: {m['Precision']:.4f}")
    report.append(f"Recall:    {m['Recall/TPR']:.4f}")
    report.append(f"FPR:       {m['FPR']:.4f}")
    report.append(f"TNR:       {m['TNR']:.4f}")
    report.append(f"FNR:       {m['FNR']:.4f}")
    
    report_text = "\n".join(report)
    print(report_text)
    
    if output_file:
        with open(output_file, "a") as f:
            f.write(report_text)
        print(f"  Results saved to {output_file}")

def predict_holdout():
    for i in range(1, 6):
        output_path = f"models\\holdout\\model{i}_output.txt"
        data_path = f"data\\holdout\\test.scale"

        model_path = f"models\\holdout\\model{i}.txt"

        cmd = f"svm-predict {data_path} {model_path} {output_path}"
        subprocess.run(cmd, shell=True)
        
        _, y_true = load_svmlight_file(data_path)
        with open(output_path, "r") as f:
            y_pred = [float(line.strip()) for line in f]
            
            metrics = calc_metrics(y_true, y_pred)
            result_file = os.path.join("models\holdout", f"model_results.txt")
            print_metrics(metrics, "Holdout Test Set Evaluation", result_file)

def predict_cv():
    for i in range(1, 6):
        model_dir = f'models\\cross_validation\\model{i}'
        fold_metrics = []

        for j in range(1, 6):
            model_path = f"{model_dir}\\model{i}_fold{j}.txt"
            data_path = f"data\\cross_validation\\val_fold{j}.scale"

            output_path = f"models\cross_validation\model{i}\model{i}_fold{j}_output.txt"
            cmd = f"svm-predict {data_path} {model_path} {output_path}"
            subprocess.run(cmd, shell=True)

            _, y_true = load_svmlight_file(data_path)
            
            with open(output_path) as f:
                y_pred = [float(line.strip()) for line in f]
            


            metrics = calc_metrics(y_true, y_pred)      
            fold_metrics.append(metrics)

        avg_metrics = {}
        for key in fold_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in fold_metrics) / len(fold_metrics)        
        
        result_file = os.path.join("models\cross_validation", "model_results.txt")
        print_metrics(avg_metrics, "Cross validation Test Set Evaluation", result_file)
            


predict_cv()