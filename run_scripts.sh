#!/bin/bash

# 繰り返し回数を指定（デフォルト: 1回）
repeat_count=${1:-1}  # コマンドライン引数があればそれを使い、なければ1回

# 繰り返し処理
for ((i=1; i<=repeat_count; i++)); do
    echo "=== Execution Round $i ==="

    # スクリプトの実行
    cd /home/sakakibara/SNNs_LSM_stdp_python/cuda/NARMA10
    python narma.py

    cd /home/sakakibara/SNNs_LSM_stdp_python/cuda/lsm
    python lsm.py

    cd /home/sakakibara/SNNs_LSM_stdp_python/cuda/tensorflow
    python MLPR_balance.py

    echo "=== Finished Round $i ==="
done

echo "=== All executions completed ==="
