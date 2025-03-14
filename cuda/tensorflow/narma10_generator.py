import numpy as np
import pandas as pd

def generate_narma10_data(steps=50000, dt=0.01, input_range=(0, 0.5), seed=42):
    """
    NARMA10データを生成する関数。
    
    Parameters:
        steps (int): データのステップ数。
        dt (float): 時間刻み。
        input_range (tuple): 入力信号の範囲 (min, max)。
        seed (int): 乱数シード（再現性のため）。
    
    Returns:
        pd.DataFrame: 時間、入力、出力を含むデータフレーム。
    """
    # 入力信号を生成
    # np.random.seed(seed)
    u = np.random.uniform(input_range[0], input_range[1], steps)
    
    # NARMA10を計算
    def narma10(input_signal):
        n = len(input_signal)
        y = np.zeros(n)
        for t in range(10, n):
            y[t] = (
                0.3 * y[t-1] +
                0.05 * y[t-1] * np.sum(y[t-10:t]) +
                1.5 * input_signal[t-10] * input_signal[t-1] +
                0.1
            )
        return y
    
    narma_output = narma10(u)
    
    # 時間軸を作成
    time = np.arange(0, steps * dt, dt)
    
    # データをデータフレームに格納
    data = pd.DataFrame({'Time': time, 'Input': u, 'Output': narma_output})
    return data

def save_narma10_data(filename='narma10_data.csv', steps=50000, dt=0.01, input_range=(0, 0.5), seed=42):
    """
    NARMA10データをCSVファイルに保存する関数。
    
    Parameters:
        filename (str): 保存するCSVファイルの名前。
        steps (int): データのステップ数。
        dt (float): 時間刻み。
        input_range (tuple): 入力信号の範囲 (min, max)。
        seed (int): 乱数シード（再現性のため）。
    """
    data = generate_narma10_data(steps, dt, input_range, seed)
    data.to_csv(filename, index=False)
    print(f"NARMA10データを '{filename}' に保存しました。")

if __name__ == "__main__":
    save_narma10_data()
