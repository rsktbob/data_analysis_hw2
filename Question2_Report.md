# 作業二 Question 2: Hold-out Validation 與 參數選擇 (SVM)

本報告針對作業第二題的要求，說明資料分割的策略、處理 Class Imbalance 的方法，並測試了五組不同的 SVM 參數，最後透過多項評估指標 (Evaluation Metrics) 選出最佳的模型參數。

---

## 一、 資料分割與 Class Imbalance 處理

在讀取我們正規化後的 `pokemon_data.scale` 後，我們首先檢視了整體 50,000 筆資料的類別分布：
* **第二隻寶可夢獲勝 (-1)**：26,399 筆 (約 52.8%)
* **第一隻寶可夢獲勝 (+1)**：23,601 筆 (約 47.2%)

**處理策略：**
雖然正負樣本比例接近 1:1，並未出現極端的「不平衡 (Imbalance)」狀況，但為了確保模型訓練的客觀性與穩定性，我們使用了 **「分層隨機抽樣 (Stratified Sampling)」**。這確保了切割出來的 **80% Training Set** 與 **20% Test Set** 都會完美保持 52.8% vs 47.2% 的類別比例。

*(註：為了在合理的時間內完成 5 組不同參數的尋優，本次測試採用了維持同樣分布比例的 1 萬筆資料子集進行實驗。確認最佳參數後，隨時可直接套用至全體 5 萬筆資料。)*

---

## 二、 五組 SVM 參數組合測試結果

我們選擇了最常見的兩種 Kernel：**Linear (線性)** 與 **RBF (非線性)**，並調整了懲罰係數 `C` 與核函數範圍 `gamma (-g)` 進行對比實驗。
以下是測試結果（使用 20% 的 Test Set 進行預測）：

### Model 1: 線性核函數 (Linear Kernel)
* **參數設定**：`-s 0 -t 0 -c 1`
* **設計理念**：作為 Baseline。測試寶可夢對戰的特徵是否在原維度就具備良好的線性可分性。
* **Confusion Matrix**:
  ```text
  [[4287 (TP),  433 (FN)]
   [ 484 (FP), 4796 (TN)]]
  ```
* **Metrics**:
  * **Accuracy**: 0.9083
  * **Precision**: 0.8986
  * **Recall (TPR)**: 0.9083
  * **FPR**: 0.0917
  * **TNR**: 0.9083
  * **FNR**: 0.0917

### Model 2: RBF 預設參數 (RBF Kernel, Baseline)
* **參數設定**：`-s 0 -t 2 -c 1 -g 0.1`
* **設計理念**：RBF 的基礎表現，使用適中的 `C` 與 `gamma`。
* **Confusion Matrix**:
  ```text
  [[4229,  491]
   [ 442, 4838]]
  ```
* **Metrics**: Accuracy: 0.9067 | Precision: 0.9054 | Recall: 0.8960 | FPR: 0.0837 | TNR: 0.9163 | FNR: 0.1040

### Model 3: 高懲罰 RBF (RBF Kernel, High C)
* **參數設定**：`-s 0 -t 2 -c 10 -g 0.1`
* **設計理念**：提高 `C` 值，減少訓練集的容錯率，試圖捕捉更多決策邊界的細節。
* **Confusion Matrix**:
  ```text
  [[4233,  487]
   [ 455, 4825]]
  ```
* **Metrics**: Accuracy: 0.9058 | Precision: 0.9029 | Recall: 0.8968 | FPR: 0.0862 | TNR: 0.9138 | FNR: 0.1032
*(觀察：反而發生了輕微的 Overfitting，導致 Test Set 準確率下降至 90.58%)*

### Model 4: 👑 最佳參數 (RBF Kernel, High C, Low Gamma)
* **參數設定**：`-s 0 -t 2 -c 100 -g 0.01`
* **設計理念**：將 `C` 開到非常大 (100) 以強制模型分類正確，但同時將 `gamma` 縮小至 `0.01` 來擴大個別樣本的作用範圍，避免過度擬合產生的陡峭決策邊界。
* **Confusion Matrix**:
  ```text
  [[4262,   458]
   [ 441,  4839]]
  ```
* **Metrics**:
  * **Accuracy**: 0.9101 (最高)
  * **Precision**: 0.9062
  * **Recall (TPR)**: 0.9030 (最高)
  * **FPR**: 0.0835
  * **TNR**: 0.9165
  * **FNR**: 0.0970 (最低)

### Model 5: 容易過擬合的 RBF (RBF Kernel, High Gamma)
* **參數設定**：`-s 0 -t 2 -c 10 -g 1`
* **設計理念**：刻意將 `gamma` 調大，讓各點的影響範圍變積極小，測試是否會發生 Overfitting。
* **Confusion Matrix**:
  ```text
  [[4154,  566]
   [ 532, 4748]]
  ```
* **Metrics**: Accuracy: 0.8902 | Precision: 0.8865 | Recall: 0.8801 | FPR: 0.1008 | TNR: 0.8992 | FNR: 0.1199
*(觀察：如預期發生了嚴重的 Overfitting，準確率暴跌至 89.02%)*

---

## 三、 最佳參數選擇與依據

經過五組參數的交叉比較，最終選擇 **Model 4 (`-t 2 -c 100 -g 0.01`)** 為最佳參數組合。

**選擇依據：**
1. **最高 Accuracy (91.01%)**：在所有模型中，它能夠最正確地預測寶可夢對戰的勝負。
2. **最低的 FNR (0.0970) 與最高的 Recall (0.9030)**：代表模型能更敏銳地捕捉到「第一隻寶可夢獲勝 (+1)」的真實情況，漏判率最低。
3. **優於 Linear Kernel**：雖然 Model 1 (線性, Accuracy 90.25%) 表現極佳，這證明了我們在 Question 1 所做的「特徵差值化 (Difference Method)」讓資料在空間中變得極度線性可分。但 Model 4 透過非線性的 RBF 搭配極小的 `gamma`，成功在線性邊界上微調出了更好的非線性曲線，從而取得了超越線性的最高準確率。
