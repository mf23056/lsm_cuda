import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# 1. LSMのスパイクデータをCSVから読み込む
spike_data = pd.read_csv('../lsm/spike_train.csv', header=None)  # スパイクデータ (時間 vs ニューロン)
target_data = pd.read_csv('../NARMA10/narma10_data.csv', header=0)  # NARMAモデルの出力（目標ラベル）

# 2. 目標データからOutput列を取り出す
y = target_data['Output'].to_numpy()

# スパイクデータ (特徴量) の転置
spike_data = spike_data.T

# 3. データを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(spike_data, y, test_size=0.2, random_state=42)

# 4. 特徴量を標準化
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 目標データを標準化
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# 5. MLP回帰モデルの訓練
mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=100, random_state=42)
mlp_model.fit(X_train_scaled, y_train_scaled.ravel())

# 6. 最終エポックの予測結果
y_pred_train_scaled = mlp_model.predict(X_train_scaled)
y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1))

y_pred_test_scaled = mlp_model.predict(X_test_scaled)
y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1))

# 7. テストデータでのMSE計算
mse_test = mean_squared_error(y_test, y_pred_test)
print(f"Test Data MSE: {mse_test}")

# 追加：MSE を既存のCSVに追加保存
csv_filename = "mse_result_balance.csv"
mse_results = pd.DataFrame({"MSE": [mse_test]})  # DataFrame に変換

# CSVファイルが存在するか確認し、適切なモードで書き込む
try:
    with open(csv_filename, "r") as f:
        header = False  # ファイルが存在する場合はヘッダーを追加しない
except FileNotFoundError:
    header = True  # 初回実行時はヘッダーを追加

# 追記（append）モードで保存
mse_results.to_csv(csv_filename, mode="a", header=header, index=False)
print(f"MSE results appended to {csv_filename}")

# 8. 訓練データとテストデータの予測結果を並べてプロット
plt.figure(figsize=(10, 4))
plt.plot(range(len(y_train[:100])), y_train[:100], label='Actual (Train)', marker='o', linestyle='--', color='blue')
plt.plot(range(len(y_train[:100])), y_pred_train[:100], label='Predicted (Train)', marker='x', linestyle='-', color='cyan')
plt.plot(range(len(y_train[:100]), len(y_train[:100]) + len(y_test[:100])), y_test[:100], label='Actual (Test)', marker='o', linestyle='--', color='red')
plt.plot(range(len(y_train[:100]), len(y_train[:100]) + len(y_test[:100])), y_pred_test[:100], label='Predicted (Test)', marker='x', linestyle='-', color='orange')
plt.axvline(x=len(y_train[:100]), color='black', linestyle='dashed', label='Train/Test Split')
plt.legend()
plt.title("Actual vs Predicted (MLP) for Training and Test Data")
plt.xlabel("Sample Index")
plt.ylabel("Output Value")
plt.grid()
plt.show()
plt.close()  # ウィンドウを開かずに閉じる
