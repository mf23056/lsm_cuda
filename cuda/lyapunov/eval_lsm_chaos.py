import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class LIF:
    def __init__(self, dt=0.01, tau_m=20, V_rest=-65.0, R=10.0, I_back=5.0, V_reset=-65.0, V_th=-50.0, V_ref=3):
        self.dt = dt
        self.tau_m = tau_m
        self.V_rest = V_rest
        self.R = R
        self.I_back = I_back
        self.V_reset = V_reset
        self.V_th = V_th
        self.V_ref_steps = int(V_ref / dt)

    def __call__(self, I_syn, before_V, ref_time):
        ref_time = torch.clamp(ref_time - 1, min=0)
        V = torch.where(ref_time > 0, torch.full_like(before_V, self.V_reset),
                        before_V + self.dt * ((1 / self.tau_m) * (-(before_V - self.V_rest) + self.R * (I_syn + self.I_back))))
        spiked = (V >= self.V_th).float()
        ref_time = torch.where(spiked > 0, self.V_ref_steps, ref_time)
        V = torch.where(spiked > 0, torch.full_like(V, self.V_reset), V)
        return spiked, V, ref_time

class StaticSynapse:
    def __init__(self, dt=0.01, tau_syn=25):
        self.dt = dt
        self.tau_syn = tau_syn

    def __call__(self, bin_spike, W, before_I):
        spikes = bin_spike.unsqueeze(1)
        return before_I + self.dt * (-before_I / self.tau_syn) + W * spikes

class LSM:
    def __init__(self, n_exc, n_inh, dt=0.01, device='cuda', weight_file=None):
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_total = n_exc + n_inh
        self.dt = dt
        self.device = device
        self.C = {"EE": 0.121, "EI": 0.169, "IE": 0.127, "II": 0.097}
        
        self.neuron = LIF()
        self.synapse = StaticSynapse()
        self._initialize_neurons(n_exc, n_inh)
        self._initialize_synapses(weight_file)

    def _initialize_neurons(self, n_exc, n_inh):
        self.sum_I_syn = torch.zeros(self.n_total, device=self.device)
        self.before_V = torch.full((self.n_total,), -65.0, device=self.device)
        self.ref_time = torch.zeros(self.n_total, device=self.device)
        self.spike_state = torch.zeros(self.n_total, device=self.device)

    def _initialize_synapses(self, weight_file):
        self.before_I = torch.zeros((self.n_total, self.n_total), device=self.device)
        if weight_file:
            weights_df = pd.read_csv(weight_file, header=None)
            print(f"Shape of weights: {weights_df.shape}")
            weights_matrix = torch.tensor(weights_df.values, dtype=torch.float32, device=self.device)
            weight_sum = weights_df.to_numpy().sum()
            print(f'The sum of the weights is: {weight_sum}')
            self.weights = weights_matrix
        else:
            self.weights = torch.randn(self.n_total, self.n_total, device=self.device)

    def run_simulation(self, inputs, calc_lyapunov=False):
        T = inputs.size(0)
        self.spike_record = torch.zeros((self.n_total, T), device=self.device)
        num_steps = T
        self.exc_input_log = torch.zeros(num_steps, device=self.device)
        self.inh_input_log = torch.zeros(num_steps, device=self.device)
        self.ei_diff_log = torch.zeros(num_steps, device=self.device)
        random_indices = torch.randint(0, self.n_total, (200,), device=self.device)

        if calc_lyapunov:
            delta = torch.randn(self.n_total, device=self.device) * 1e-5
            delta /= torch.norm(delta)
            lyapunov_exponents = torch.zeros(T, device=self.device)

        for t in range(1, T):
            self.before_I = self.synapse(self.spike_state, self.weights, self.before_I)
            self.sum_I_syn = torch.sum(self.before_I, dim=0)
            self.sum_I_syn[random_indices] += inputs[t]
            self.spike_state, self.before_V, self.ref_time = self.neuron(self.sum_I_syn, self.before_V, self.ref_time)

            EI_input = torch.sum(self.before_I, dim=1)
            exc_input = torch.sum(EI_input[:self.n_exc])
            inh_input = torch.sum(EI_input[self.n_exc:])
            ei_diff = exc_input + inh_input

            self.exc_input_log[t] = exc_input
            self.inh_input_log[t] = inh_input
            self.ei_diff_log[t] = ei_diff
            self.spike_record[:, t] = self.spike_state

            if calc_lyapunov:
                perturbed_V = self.before_V + delta
                _, perturbed_V, _ = self.neuron(self.sum_I_syn, perturbed_V, self.ref_time)
                delta = perturbed_V - self.before_V
                delta_norm = torch.norm(delta)
                delta_norm = max(delta_norm, 1e-10)  # 最小値を設定
                lyapunov_exponents[t] = torch.log(torch.tensor(delta_norm, device=self.device))  # Tensorに変換
                delta /= delta_norm

        if calc_lyapunov:
            self.lyapunov_exponents = lyapunov_exponents
            self.average_lyapunov = torch.mean(lyapunov_exponents[1:])  # 最初を除外して平均
            print(f"Average Lyapunov Exponent: {self.average_lyapunov.item()}")


    def save_spikes_to_csv(self, filename="spike_train.csv"):
        spike_train = self.spike_record.cpu().numpy()
        np.savetxt(filename, spike_train, delimiter=",")
        print(f"Spikes saved to {filename}")

    def plot_raster(self):
        spike_times = torch.nonzero(self.spike_record, as_tuple=False)
        plt.figure(figsize=(12, 8))
        plt.scatter(spike_times[:, 1].cpu() * self.dt, spike_times[:, 0].cpu(), marker="|", color="black", s=10)
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron Index")
        plt.title("Spike Raster Plot")
        plt.savefig("spike_raster_lsm.png", dpi=300)
        plt.show()


if __name__ == '__main__':
    scale_factor = 500
    num_steps = 10000  # Number of time steps

    # Generate random noise as input
    torch.manual_seed(42)
    random_input = torch.randn(num_steps, dtype=torch.float32) * scale_factor
    weight_file = "../random_netwrok/weights_15_15_20_25_25_07_10_15.csv"
    # weight_file = '../random_netwrok/weights.csv'
    # weight_file = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    random_input = random_input.to(device)

    network = LSM(n_exc=1000, n_inh=250, device=device, weight_file=weight_file)
    spike_record = network.run_simulation(random_input, calc_lyapunov=None)
    network.plot_raster()
    network.save_spikes_to_csv()

