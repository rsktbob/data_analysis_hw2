import numpy as np
import matplotlib.pyplot as plt

# 讀機率輸出
with open('best_model_output.txt', 'r') as f:
    lines = f.read().splitlines()

header = lines[0].split()
labels_order = [int(x) for x in header[1:]]
prob_pos_idx = labels_order.index(1)

predict_data = [line.split() for line in lines[1:]]
y_prob = [float(row[1 + prob_pos_idx]) for row in predict_data]

# 讀真實的標籤
with open('data\\holdout\\test.scale', 'r') as f:
    lines = f.read().splitlines()

y_true = np.array([int(line.split()[0]) for line in lines])

# 手動計算 ROC
def calc_roc(y_true, y_prob):
    thresholds = sorted(set(y_prob), reverse=True)

    P = np.sum(y_true == 1)
    N = np.sum(y_true == -1)
    
    tprs = [0.0]
    fprs = [0.0]
    
    for thresh in thresholds:
        y_pred = np.where(np.array(y_prob) >= thresh, 1, -1)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == -1))
        tprs.append(TP / P)
        fprs.append(FP / N)
    
    tprs.append(1.0)
    fprs.append(1.0)
    return np.array(fprs), np.array(tprs)

# 手動計算 AUC（梯形法）
def calc_auc(fprs, tprs):
    auc = 0.0
    for i in range(1, len(fprs)):
        dx = fprs[i] - fprs[i-1]
        avg_y = (tprs[i] + tprs[i-1]) / 2
        auc += dx * avg_y
    return auc

fprs, tprs = calc_roc(y_true, y_prob)
auc = calc_auc(fprs, tprs)

print(f"AUC: {auc:.4f}")

plt.figure(figsize=(7, 6))
plt.plot(fprs, tprs, color='blue', label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()