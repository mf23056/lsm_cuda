import torch
import matplotlib.pyplot as plt
import math
import random
import numpy as np

class LIF:
    def __init__(self, dt=0.01, tau_m=30, V_rest=0.0, R=1.0, I_back=15.5, V_reset=13.5, V_th=15.0, V_ref=3):
        '''
        param dt: 刻み幅 [ms]
        param tau_m: 膜時間定数 [ms]
        param V_rest: 静止電位 [mV]
        param R: 抵抗 [MΩ]
        param I_back: バックグラウンド電流 [nA]
        param V_reset: リセット電位 [mV]
        param V_th: 閾値電位 [mV]
        param V_ref: 不応期 [ms]
        '''
        self.dt = dt
        self.tau_m = tau_m
        self.V_rest = V_rest
        self.R = R
        self.I_back = I_back
        self.V_reset = V_reset
        self.V_th = V_th
        self.V_ref_steps = int(V_ref / dt)

    def __call__(self, I_syn, before_V, ref_time):
        # 不応期のカウントダウン
        ref_time = torch.clamp(ref_time - 1, min=0)

        # 膜電位の更新
        V = torch.where(ref_time > 0, torch.full_like(before_V, self.V_reset),
                        before_V + self.dt * ((1 / self.tau_m) * (-(before_V - self.V_rest) + self.R * (I_syn + self.I_back))))

        # スパイク判定
        spiked = (V >= self.V_th).float()

        # スパイクが発生したニューロンの処理
        ref_time = torch.where(spiked > 0, self.V_ref_steps, ref_time)
        V = torch.where(spiked > 0, torch.full_like(V, self.V_reset), V)
        return spiked, V, ref_time


class StaticSynapse:
    def __init__(self, dt=0.01, tau_syn=40):
        self.dt = dt
        self.tau_syn = tau_syn

    def __call__(self, bin_spike, W, before_I):
        # bin_spike は1か0のバイナリスパイク列
        spikes = bin_spike.unsqueeze(1)  # 各スパイクをテンソルに合わせて繰り返す

        return before_I + self.dt * (-before_I / self.tau_syn) + W * spikes


class Guetig_STDP:
    def __init__(self, dt=0.01, A_plus=0.01, A_minus=0.01, tau_plus=30.0, tau_minus=30.0, alpha=0.90, device='cuda'):
        self.dt = dt
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.alpha = alpha
        self.device = device

    def __call__(self, delta_t):
        delta_w = torch.where(
            delta_t > 0,
            self.A_plus
            * (torch.exp(-delta_t / self.tau_plus) - self.alpha * torch.exp(-delta_t / (self.alpha * self.tau_plus))),
            
            -self.A_minus
            * (torch.exp(delta_t / self.tau_minus) - self.alpha * torch.exp(delta_t / (self.alpha * self.tau_minus))),
        )
        
       
        return delta_w


class SNN:
    def __init__(self, n_exc, n_inh, dt=0.01, device='cuda'):
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_total = n_exc + n_inh
        self.dt = dt
        self.device = device
        
        #####
        self.E_A_plus=0.01
        self.E_A_minus=0.01
        self.E_tau_plus=20.0
        self.E_tau_minus=40.0
        self.E_alpha=0.90

        self.I_A_plus=0.018
        self.I_A_minus=0.008
        self.I_tau_plus=30.0
        self.I_tau_minus=30.0
        self.I_alpha=0.90


        ####plot
        # パラメータを右側に表示
        self.param_str = f"""
        E_A_plus = {self.E_A_plus}
        E_A_minus = {self.E_A_minus}
        E_tau_plus = {self.E_tau_plus}
        E_tau_minus = {self.E_tau_minus}
        E_alpha = {self.E_alpha}

        I_A_plus = {self.I_A_plus}
        I_A_minus = {self.I_A_minus}
        I_tau_plus = {self.I_tau_plus}
        I_tau_minus = {self.I_tau_minus}
        I_alpha = {self.I_alpha}
        """
        
        """
        このままで

        総シナプス接続数: 194741 (12.47%)
        興奮性ニューロン → 興奮性ニューロンのシナプス接続数: 117966 (60.58%)
        興奮性ニューロン → 抑制性ニューロンのシナプス接続数: 40092 (20.59%)
        抑制性ニューロン → 興奮性ニューロンのシナプス接続数: 30682 (15.76%)
        抑制性ニューロン → 抑制性ニューロンのシナプス接続数: 6001 (3.08%)
        """

        self.neuron = LIF()
        self.synapse = StaticSynapse()
        self.E_stdp = Guetig_STDP(A_plus=self.E_A_plus, 
                                  A_minus=self.E_A_minus, 
                                  tau_plus=self.E_tau_plus, 
                                  tau_minus=self.E_tau_minus, 
                                  alpha=self.E_alpha, device=self.device)
        self.I_stdp = Guetig_STDP(A_plus=self.I_A_plus, 
                                  A_minus=self.I_A_minus, 
                                  tau_plus=self.I_tau_plus,
                                  tau_minus=self.I_tau_minus, 
                                  alpha=self.I_alpha,device=self.device)

        self._initialize_neurons(n_exc, n_inh)
        self._initialize_synapses()
        # self.plot_synapse_weights()

    def _initialize_neurons(self, n_exc, n_inh):
        self.sum_I_syn = torch.zeros(self.n_total, device=self.device)
        self.before_V = torch.full((self.n_total,), 0.0, device=self.device)
        self.ref_time = torch.zeros(self.n_total, device=self.device)
        self.spike_state = torch.zeros(self.n_total, device=self.device)
      


    def _initialize_synapses(self):
        self.before_I = torch.zeros((self.n_total, self.n_total), device=self.device)
        self.weights = torch.rand((self.n_total, self.n_total), device=self.device)
        self.weights[:self.n_exc, :] *= 0.1
        self.weights[self.n_exc:,:] *= -0.1 # randomでもItoE, ItoIはマイナス
        # self.weights[:self.n_exc, :] *= 0.8
        # self.weights[self.n_exc:, :] *= -0.3
        # self.weights[:self.n_exc, :self.n_exc] *= 0.023
        # self.weights[:self.n_exc, self.n_exc:] *= 0.005
        # self.weights[self.n_exc:, :self.n_exc] *= -0.08
        # self.weights[self.n_exc:, self.n_exc:] *= -0.02
        self.weights.fill_diagonal_(0)

        self.weight_bias = torch.zeros((self.n_total, self.n_total), device=self.device)
        self.weight_bias[:self.n_exc, :] = 1
        self.weight_bias[self.n_exc:, :] = -1


    def _count_synapses(self):
        # 各タイプのシナプスの接続が存在する場所を数える
        exc_to_exc = torch.sum(self.weights[:self.n_exc, :self.n_exc] != 0).item()  # E → E のシナプス接続数
        exc_to_inh = torch.sum(self.weights[:self.n_exc, self.n_exc:] != 0).item()  # E → I のシナプス接続数
        inh_to_exc = torch.sum(self.weights[self.n_exc:, :self.n_exc] != 0).item()  # I → E のシナプス接続数
        inh_to_inh = torch.sum(self.weights[self.n_exc:, self.n_exc:] != 0).item()  # I → I のシナプス接続数

        # 総シナプス接続数
        total_connections = torch.sum(self.weights != 0).item()

        # 仮にすべてのニューロンが結合している場合の接続数
        total_possible_connections = self.n_total * self.n_total - self.n_total  # 自己結合を除く

        # 各シナプスタイプの割合（接続数ベース）
        exc_to_exc_ratio = exc_to_exc / total_connections
        exc_to_inh_ratio = exc_to_inh / total_connections
        inh_to_exc_ratio = inh_to_exc / total_connections
        inh_to_inh_ratio = inh_to_inh / total_connections

        # 結果の表示
        print(f"総シナプス接続数: {total_connections} ({total_connections/total_possible_connections * 100:.2f}%)")
        print(f"興奮性ニューロン → 興奮性ニューロンのシナプス接続数: {exc_to_exc} ({exc_to_exc_ratio * 100:.2f}%)")
        print(f"興奮性ニューロン → 抑制性ニューロンのシナプス接続数: {exc_to_inh} ({exc_to_inh_ratio * 100:.2f}%)")
        print(f"抑制性ニューロン → 興奮性ニューロンのシナプス接続数: {inh_to_exc} ({inh_to_exc_ratio * 100:.2f}%)")
        print(f"抑制性ニューロン → 抑制性ニューロンのシナプス接続数: {inh_to_inh} ({inh_to_inh_ratio * 100:.2f}%)")

    def run_simulation(self, T=1000):
        now_spike_time = torch.zeros(self.n_total, device=self.device)
        self.spike_record = torch.zeros((self.n_total, int(T / self.dt)), device=self.device)
        
        # ログ用テンソルを初期化
        num_steps = int(T / self.dt)
        self.exc_input_log = torch.zeros(num_steps, device=self.device)
        self.inh_input_log = torch.zeros(num_steps, device=self.device)
        self.ei_diff_log = torch.zeros(num_steps, device=self.device)

        # 重み変化を記録するテンソルを初期化
        # 重み変化を記録するテンソルを初期化
        num_steps = int(T / self.dt)
        self.weight_log = torch.zeros((num_steps, 4), device=self.device)


        self.weight_clip = self.weights.clone()


        for t in range(1, int(T / self.dt)):
            # print(self.spike_state)
            self.before_I = self.synapse(self.spike_state, self.weights, self.before_I)
            self.sum_I_syn = torch.sum(self.before_I, dim=0)
            self.spike_state, self.before_V, self.ref_time = self.neuron(self.sum_I_syn, self.before_V, self.ref_time)
            
            # STDPの処理（省略なし）
            if torch.any(self.spike_state == 1):
                now_spike_time[self.spike_state == 1] = t 
                delta_t = now_spike_time.unsqueeze(0) - now_spike_time.unsqueeze(1) #ok
                
                E_delta_w = self.E_stdp(delta_t[:self.n_exc, :]) #ok
                I_delta_w = self.I_stdp(delta_t[self.n_exc:, :]) #ok
                delta_w = torch.cat((E_delta_w, I_delta_w), dim=0) #ok
                delta_w[self.n_exc:, :] *= -1 #ok

                # 自身の条件を満たさない箇所を 0 にする
                row_mask = self.spike_state != 1  # 行のマスク
                col_mask = self.spike_state != 1  # 列のマスク

                # 外積を用いて行列マスクを作成
                matrix_mask = row_mask.unsqueeze(1) & col_mask.unsqueeze(0)  # 行×列のブールマスク

                # delta_wの該当箇所を0に設定
                delta_w[matrix_mask] = 0
                
                # print(torch.nonzero(delta_w, as_tuple=True))
                self.weights += delta_w 
                self.weights[:self.n_exc, :] = torch.maximum(self.weights[:self.n_exc, :], torch.zeros_like(self.weights[:self.n_exc, :], device=self.device))
                self.weights[self.n_exc:, :] = torch.minimum(self.weights[self.n_exc:, :], torch.zeros_like(self.weights[self.n_exc:, :], device=self.device))

            # 重みの集計 (E-E, E-I, I-E, I-I)
            ee_sum = self.weights[:self.n_exc, :self.n_exc].sum()
            ei_sum = self.weights[:self.n_exc, self.n_exc:].sum()
            ie_sum = self.weights[self.n_exc:, :self.n_exc].sum()
            ii_sum = self.weights[self.n_exc:, self.n_exc:].sum()
            self.weight_log[t, :] = torch.tensor([ee_sum, ei_sum, ie_sum, ii_sum], device=self.device)
            
            # Excitatory (興奮性) と Inhibitory (抑制性) の入力を分けてログ
            EI_input = torch.sum(self.before_I, dim = 1)
            exc_input = torch.sum(EI_input[:self.n_exc])
            inh_input = torch.sum(EI_input[self.n_exc:])
            ei_diff = exc_input + inh_input  # E - I の差を計算

            # GPU 上でログに保存
            self.exc_input_log[t] = exc_input
            self.inh_input_log[t] = inh_input
            self.ei_diff_log[t] = ei_diff
            self.spike_record[:, t] = self.spike_state

        self.weights_now = self.weights
        changed_weights = self.weights_now - self.weight_clip
        if torch.any(changed_weights != 0):
            print("0 じゃない部分があります")
            print(self.weight_log.sum())
            print('変化量',changed_weights.sum())
        else:
            print("すべて 0 です")

        
        # 固有値を計算
        eigenvalues = torch.linalg.eigvals(self.weights)

        # 最大固有値を求める
        max_eigenvalue = torch.max(eigenvalues.abs())

        print(max_eigenvalue.item())

        return self.spike_record

    def plot_raster(self):
        spike_times = torch.nonzero(self.spike_record, as_tuple=False)
        plt.figure(figsize=(12, 8))
        plt.scatter(spike_times[:, 1].cpu() * self.dt, spike_times[:, 0].cpu(), marker="|", color="black", s=10)
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron Index")
        plt.title("Spike Raster Plot")
        
        plt.text(0.98, 0.5, self.param_str, transform=plt.gca().transAxes, fontsize=8, verticalalignment='center')
         # 右側の余白を広げる
        plt.subplots_adjust(right=0.88)  # 右側の余白を広げる
        plt.savefig("spike_raster.png", dpi=300)
        plt.show()

    def plot_inputs(self):
        # ログを CPU に転送
        exc_input_log = self.exc_input_log.cpu().numpy()
        inh_input_log = self.inh_input_log.cpu().numpy()
        ei_diff_log = self.ei_diff_log.cpu().numpy()

        # プロット
        plt.figure(figsize=(12, 8))
        time = np.arange(len(exc_input_log)) * self.dt
        plt.plot(time, exc_input_log, label='Excitatory PSP', color='b')
        plt.plot(time, inh_input_log, label='Inhibitory PSP', color='r')
        plt.plot(time, ei_diff_log, label='E-I PSP Difference', color='k')
        plt.xlabel('Time (ms)')
        plt.ylabel('Potential Sum')
        plt.title('Excitatory and Inhibitory PSP Over Time')
        plt.legend()
        plt.grid(True)

        plt.text(0.98, 0.5, self.param_str, transform=plt.gca().transAxes, fontsize=8, verticalalignment='center')
         # 右側の余白を広げる
        plt.subplots_adjust(right=0.88)  # 右側の余白を広げる
        plt.savefig("E_I_PSP.png", dpi=300)
        plt.show()

    def plot_weights(self):
        # 重み変化のログを CPU に転送
        weight_log = self.weight_log.cpu().numpy()
        
        # プロット
        plt.figure(figsize=(12, 8))
        time = np.arange(weight_log.shape[0]) * self.dt
        plt.plot(time, weight_log[:, 0], label='E-E', color='b')
        plt.plot(time, weight_log[:, 1], label='E-I', color='r')
        plt.plot(time, weight_log[:, 2], label='I-E', color='g')
        plt.plot(time, weight_log[:, 3], label='I-I', color='k')
        plt.xlabel('Time (ms)')
        plt.ylabel('Sum of Synaptic Weight')
        plt.title('Synaptic Weight Changes Over Time')
        plt.legend()
        plt.grid(True)

        plt.text(0.98, 0.5, self.param_str, transform=plt.gca().transAxes, fontsize=8, verticalalignment='center')
         # 右側の余白を広げる
        plt.subplots_adjust(right=0.88)  # 右側の余白を広げる
        plt.savefig("weight_changes.png", dpi=300)
        plt.show()

    def plot_synapse_weights(self):
        plt.figure(figsize=(12, 8))

        # シナプス接続の重みをCPUに転送してからプロット
        weights_cpu = self.weights.cpu()  # GPUからCPUに転送

        # シナプス接続がある場所のみプロット
        for i in range(self.n_total):
            for j in range(self.n_total):
                if weights_cpu[i, j] != 0:
                    plt.plot(i, j, 'k.', markersize=1)  # ドットで表示

        plt.xlabel("Pre-neuron index")
        plt.ylabel("Post-neuron index")
        plt.title("Synaptic Connections")
        plt.grid(True)
        plt.savefig("synapse_connections.png", dpi=300)
        plt.show()

    def save_weights_to_csv(self, filename="weights.csv"):
        # GPUからCPUに転送し、numpy配列に変換
        weights_np = self.weights.cpu().numpy()
        # CSVファイルとして保存
        np.savetxt(filename, weights_np, delimiter=",")
        print(f"Weights saved to {filename}")


if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    network = SNN(n_exc=1000, n_inh=250, device=device)
    T = 4000  # ms
    spike_record = network.run_simulation(T)
    network.plot_raster()
    network.plot_inputs()
    network.plot_weights()

    # 保存
    network.save_weights_to_csv("weights.csv")




    
