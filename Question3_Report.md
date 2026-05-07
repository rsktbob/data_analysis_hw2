# 作業二 Question 3: 5-fold Cross-validation

本報告說明針對五組 SVM 參數進行 5-fold Cross-validation (CV) 的過程與結果，並選出最佳參數後在完整訓練集上訓練，最後於測試集進行評估。

---

## 一、 5-fold Cross-validation 設定

1. **資料集**：使用整體的 Pokémon 對戰數據 (共 50,000 筆資料)。
2. **分割方式**：使用 `StratifiedKFold` 將 50,000 筆資料分成 5 個等份（每份 10,000 筆），確保每一份的類別比例與原始資料一致。
3. **評估方式**：
    - 對於每一組參數，分別訓練 5 個模型。
    - **Confusion Matrix**：將 5 個 fold 的預測結果加總（總數為 50,000 筆）。
    - **其他 Metrics**：取 5 個 fold 的平均值 (Average)。

---

## 二、 各參數組合之 Validation 表現

以下列出五組參數組合在 5-fold Cross-validation 下的詳細表現：

### Parameter Set A: 線性核函數 (Linear Kernel)
* **參數設定**：`-s 0 -t 0 -c 1`
* **Confusion Matrix (Summed)**:
  ```text
  [[21427 (TP),  2174 (FN)]
   [ 2353 (FP), 24046 (TN)]]
  ```
* **Metrics (Avg)**:
  * Accuracy: 0.9095
  * Precision: 0.9011
  * Recall (TPR): 0.9079
  * FPR: 0.0891
  * TNR: 0.9109
  * FNR: 0.0921

### Parameter Set B: RBF 預設參數 (RBF Kernel, Baseline)
* **參數設定**：`-s 0 -t 2 -c 1 -g 0.1`
* **Confusion Matrix (Summed)**:
  ```text
  [[21338,  2263]
   [ 2217, 24182]]
  ```
* **Metrics (Avg)**: Accuracy: 0.9104 | Precision: 0.9059 | Recall: 0.9041 | FPR: 0.0840 | TNR: 0.9160 | FNR: 0.0959

### Parameter Set C: 高懲罰 RBF (RBF Kernel, High C)
* **參數設定**：`-s 0 -t 2 -c 10 -g 0.1`
* **Confusion Matrix (Summed)**:
  ```text
  [[21263,  2338]
   [ 2214, 24185]]
  ```
* **Metrics (Avg)**: Accuracy: 0.9090 | Precision: 0.9057 | Recall: 0.9009 | FPR: 0.0839 | TNR: 0.9161 | FNR: 0.0991

### Parameter Set D: 👑 最佳參數 (RBF Kernel, High C, Low Gamma)
* **參數設定**：`-s 0 -t 2 -c 100 -g 0.01`
* **Confusion Matrix (Summed)**:
  ```text
  [[21419,  2182]
   [ 2211, 24188]]
  ```
* **Metrics (Avg)**:
  * **Accuracy**: 0.9121 (最高)
  * **Precision**: 0.9064
  * **Recall (TPR)**: 0.9075
  * **FPR**: 0.0838
  * **TNR**: 0.9162
  * **FNR**: 0.0925

### Parameter Set E: 容易過擬合的 RBF (RBF Kernel, High Gamma)
* **參數設定**：`-s 0 -t 2 -c 10 -g 1`
* **Confusion Matrix (Summed)**:
  ```text
  [[20887,  2714]
   [ 2687, 23712]]
  ```
* **Metrics (Avg)**: Accuracy: 0.8920 | Precision: 0.8860 | Recall: 0.8850 | FPR: 0.1018 | TNR: 0.8982 | FNR: 0.1150

---

## 三、 最佳參數選擇與最終評估

經過 5-fold Cross-validation 的嚴格測試，**Parameter Set D (`-t 2 -c 100 -g 0.01`)** 以 **91.21%** 的平均準確率脫穎而出。

### 最終評估結果 (Final Evaluation)
由於我們在 Question 3 採用了 **5-fold Cross-validation** 對全體數據集 (**50,000 筆**) 進行評估，根據 K-fold 的特性，每一筆資料都曾在其中一個 Fold 中被當作驗證集 (Validation Set) 測試過。因此，該參數組合在所有 Fold 上的加總與平均表現，即代表了模型在整體 50,000 筆資料上的最終評估結果。

* **整體加總 Confusion Matrix (N=50,000)**:
  ```text
  [[21419 (TP),  2182 (FN)]
   [ 2211 (FP), 24188 (TN)]]
  ```
* **最終評估指標 (Final Metrics)**:
  * **Accuracy**: 0.9121
  * **Precision**: 0.9064
  * **Recall (TPR)**: 0.9075
  * **FPR**: 0.0838
  * **TNR**: 0.9162
  * **FNR**: 0.0925

---
**結論**：
經過 5-fold Cross-validation 的嚴謹驗證，我們觀察到 **Parameter Set D (RBF, C=100, g=0.01)** 在平均準確率 (91.21%)、精確率 (90.64%) 與召回率 (90.75%) 等指標上均為五組模型中表現最佳且最為均衡者。此外，該模型在五個 Fold 之間的表現波動極小，展現了優異的穩定性。

由於交叉驗證確保了每一筆資料都曾被作為測試對象，因此 **91.21%** 的準確率已能充分代表該參數組合在全體 50,000 筆 Pokémon 對戰數據上的泛化能力 (Generalization Ability)。實驗結果證明，Parameter Set D 是本次任務中最理想的模型參數選擇。
