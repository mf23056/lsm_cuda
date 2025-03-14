import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSVファイルの読み込み
mse_balance = pd.read_csv('mse_result_balance.csv')
mse_random = pd.read_csv('mse_result_random.csv')

# MSEの列名
mse_balance_column = 'MSE_balance'
mse_random_column = 'MSE_random'

# 平均と標準偏差を計算する関数
def calculate_stats(data):
    return data.mean(), data.std()

# 外れ値の検出と除去（IQRを用いた方法）
def remove_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]

# MSE ≥ 1.0 を除去
def remove_outliers_mse_threshold(data):
    return data[data < 1.0]

# データの取得
mse_balance = mse_balance[mse_balance_column].dropna()
mse_random = mse_random[mse_random_column].dropna()

# フィルタリング
mse_balance_iqr = remove_outliers_iqr(mse_balance).dropna()
mse_random_iqr = remove_outliers_iqr(mse_random).dropna()

mse_balance_mse1 = remove_outliers_mse_threshold(mse_balance).dropna()
mse_random_mse1 = remove_outliers_mse_threshold(mse_random).dropna()

# 各対象データごとに箱ひげ図を作成
for datasets, label in zip([[mse_balance, mse_random], [mse_balance_iqr, mse_random_iqr], [mse_balance_mse1, mse_random_mse1]], 
                           ["Original", "IQR", "MSE < 1.0"]):
    plt.figure(figsize=(10, 6))
    plt.boxplot(datasets, patch_artist=True, showfliers=False, widths=0.5, labels=["Balance", "Random"])
    plt.ylabel("MSE")
    plt.title(f"Boxplot of {label}")
    plt.savefig(f"../../pngs_for_paper/boxplot_{label.replace(' ', '_')}.png")
    plt.show()

# 各対象データごとにヒストグラムを作成
for datasets, label in zip([[mse_balance, mse_random], [mse_balance_iqr, mse_random_iqr], [mse_balance_mse1, mse_random_mse1]], 
                           ["Original", "IQR", "MSE < 1.0"]):
    plt.figure(figsize=(10, 6))
    plt.hist(datasets, bins=20, edgecolor="black", alpha=0.7, label=["Balance", "Random"])
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {label}")
    plt.legend()
    plt.savefig(f"../../pngs_for_paper/histogram_{label.replace(' ', '_')}.png")
    plt.show()

# 平均と標準偏差の計算と出力
print("=== Mean and Standard Deviation for Each Filter ===")
for label, data in zip(["Original", "IQR", "MSE < 1.0"], 
                        [[mse_balance, mse_random], [mse_balance_iqr, mse_random_iqr], [mse_balance_mse1, mse_random_mse1]]):
    mean_balance, std_balance = calculate_stats(data[0])
    mean_random, std_random = calculate_stats(data[1])
    print(f"{label}:")
    print(f"  Balance - Mean: {mean_balance:.6f}, Std Dev: {std_balance:.6f}")
    print(f"  Random - Mean: {mean_random:.6f}, Std Dev: {std_random:.6f}")
    print("----------------------------------")

# 平均と標準偏差のバーグラフ
means = [mse.mean() for mse in [mse_balance, mse_balance_iqr, mse_balance_mse1, mse_random, mse_random_iqr, mse_random_mse1]]
std_devs = [mse.std() for mse in [mse_balance, mse_balance_iqr, mse_balance_mse1, mse_random, mse_random_iqr, mse_random_mse1]]
dataset_labels = ["Balance (Original)", "Balance (IQR)", "Balance (MSE < 1.0)", "Random (Original)", "Random (IQR)", "Random (MSE < 1.0)"]

plt.figure(figsize=(12, 6))
plt.bar(dataset_labels, means, yerr=std_devs, capsize=5, alpha=0.7)
plt.xticks(rotation=20)
plt.ylabel("MSE")
plt.title("Mean and Standard Deviation of MSE")
plt.savefig("../../pngs_for_paper/mean_std_mse.png")
plt.show()
