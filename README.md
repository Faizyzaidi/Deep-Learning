# MS 22 — Theory Assignment 1
**Multi-Output Time-Series Forecasting using LSTM & GRU**
Jamia Millia Islamia | Roll No: 26

---

## Dataset
- **Source:** [NIFTY-50 Stock Market Data (2000–2021)](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data)
- **File used:** `IOC.csv` (Indian Oil Corporation)
- **Features (11):** Prev Close, Open, High, Low, Last, Close, VWAP, Volume, Turnover, Trades, Deliverable Volume

---

## Setup

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

Place `IOC.csv` in the same directory as the notebook, then run all cells top to bottom.

---

## Pipeline

1. **Data Cleaning** — parse dates, fix dtypes, fill NaNs (ffill/bfill), remove duplicates
2. **Split** — 80% train (2000–2017) / 20% test (2017–2021), chronological
3. **Normalize** — MinMaxScaler fit on train only
4. **Sliding Window** — input: last 10 days → output: next 5 days (step = 5, overlap = 5)
5. **Models** — LSTM and GRU (2 layers, hidden=128, dropout=0.2) via PyTorch
6. **Training** — Adam optimizer, MSE loss, early stopping (patience=15)
7. **Evaluation** — MSE, RMSE, MAE per feature + overall

---

## Results (Test Set)

| Model | Overall RMSE | Overall MAE |
|-------|-------------|------------|
| LSTM  | 4.59e+13    | 9.79e+12   |
| GRU   | 3.81e+13    | 8.49e+12   |

> GRU outperforms LSTM on this dataset. High absolute errors are driven by large-scale features (Turnover, Volume).

---

## Outputs

| File | Description |
|------|-------------|
| `loss_curves.png` | Train vs validation loss (LSTM & GRU) |
| `rmse_bar.png` | Per-feature RMSE comparison |
| `datewise_close_price.png` | Predicted vs true Close price over time |
| `datewise_volume.png` | Predicted vs true Volume over time |
| `all_features_predictions.png` | All 11 features — true vs predicted |
| `lstm_ioc_model.pth` | Saved LSTM weights |
| `gru_ioc_model.pth` | Saved GRU weights |

---

## References
- [PyTorch LSTM Docs](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [PyTorch GRU Docs](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- [Kaggle Dataset](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data)
