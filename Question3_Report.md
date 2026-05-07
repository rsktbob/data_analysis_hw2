# 作業二 Question 3: 5-fold Cross-validation

本報告說明針對五組 SVM 參數進行 5-fold Cross-validation (CV) 的過程與結果，並選出最佳參數後在完整訓練集上訓練，最後於測試集進行評估。

---

## 一、 5-fold Cross-validation 設定

1. **資料集**：沿用 Question 2 的 Training Set (共 8,000 筆資料)。
2. **分割方式**：使用 `StratifiedKFold` 將 8,000 筆資料分成 5 個等份（每份 1,600 筆），確保每一份的類別比例與原始資料一致。
3. **評估方式**：
    - 對於每一組參數，分別訓練 5 個模型。
    - **Confusion Matrix**：將 5 個 fold 的預測結果加總。
    - **其他 Metrics**：取 5 個 fold 的平均值 (Average)。

---

## 二、 各參數組合之 Validation 表現

以下列出五組參數組合在 5-fold Cross-validation 下的詳細表現：

### Parameter Set A: 線性核函數 (Linear Kernel)
* **參數設定**：`-s 0 -t 0 -c 1`
* **Confusion Matrix (Summed)**:
  ```text
  [[3403 (TP),  373 (FN)]
   [ 406 (FP), 3818 (TN)]]
  ```
* **Metrics (Avg)**:
  * Accuracy: 0.9026
  * Precision: 0.8934
  * Recall (TPR): 0.9012
  * FPR: 0.0961
  * TNR: 0.9039
  * FNR: 0.0988

### Parameter Set B: RBF 預設參數 (RBF Kernel, Baseline)
* **參數設定**：`-s 0 -t 2 -c 1 -g 0.1`
* **Confusion Matrix (Summed)**:
  ```text
  [[3324,  452]
   [ 400, 3824]]
  ```
* **Metrics (Avg)**: Accuracy: 0.8935 | Precision: 0.8926 | Recall: 0.8803 | FPR: 0.0947 | TNR: 0.9053 | FNR: 0.1197

### Parameter Set C: 高懲罰 RBF (RBF Kernel, High C)
* **參數設定**：`-s 0 -t 2 -c 10 -g 0.1`
* **Confusion Matrix (Summed)**:
  ```text
  [[3358,  418]
   [ 415, 3809]]
  ```
* **Metrics (Avg)**: Accuracy: 0.8959 | Precision: 0.8901 | Recall: 0.8893 | FPR: 0.0982 | TNR: 0.9018 | FNR: 0.1107

### Parameter Set D: 👑 最佳參數 (RBF Kernel, High C, Low Gamma)
* **參數設定**：`-s 0 -t 2 -c 100 -g 0.01`
* **Confusion Matrix (Summed)**:
  ```text
  [[3399,  377]
   [ 400, 3824]]
  ```
* **Metrics (Avg)**:
  * **Accuracy**: 0.9029 (最高)
  * **Precision**: 0.8947
  * **Recall (TPR)**: 0.9002
  * **FPR**: 0.0947
  * **TNR**: 0.9053
  * **FNR**: 0.0998

### Parameter Set E: 容易過擬合的 RBF (RBF Kernel, High Gamma)
* **參數設定**：`-s 0 -t 2 -c 10 -g 1`
* **Confusion Matrix (Summed)**:
  ```text
  [[3170,  606]
   [ 569, 3655]]
  ```
* **Metrics (Avg)**: Accuracy: 0.8531 | Precision: 0.8478 | Recall: 0.8395 | FPR: 0.1347 | TNR: 0.8653 | FNR: 0.1605

---

## 三、 最佳參數選擇與最終評估

經過 5-fold CV，**Parameter Set D (`-t 2 -c 100 -g 0.01`)** 以 **90.29%** 的平均準確率勝出。

### 最終模型訓練與測試 (Best Parameter: Set D)
我們使用最佳參數對「整個 Training Set (8,000筆)」重新進行訓練，並在「Test Set (2,000筆)」上評估。

* **Test Set Confusion Matrix**:
  ```text
  [[847 (TP),  97 (FN)]
   [ 94 (FP), 962 (TN)]]
  ```
* **Evaluation Metrics (Final)**:
  * **Accuracy**: 0.9045
  * **Precision**: 0.9001
  * **Recall (TPR)**: 0.8972
  * **FPR**: 0.0890
  * **TNR**: 0.9110
  * **FNR**: 0.1028

---
**結論**：
透過 5-fold Cross-validation，我們驗證了 Model D 的穩定性。最終在 Test Set 上取得了 90.45% 的準確率，表現與 CV 期間的平均值 (90.29%) 相當接近，證明模型具有良好的泛化能力 (Generalization Ability)。
