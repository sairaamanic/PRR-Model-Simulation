import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- 1. CONFIGURATION: The Lichtman Proxy ---
N = 100                # Number of ROI Nodes (Downsampled from H01 dataset)
K_CRITICAL = 1.8       # Predicted Coupling Threshold for "Wisdom"
TIME_SEC = 20          # Duration of simulated cognition (Trials)
DT = 0.01              # Temporal resolution (10ms)

# --- 2. CONNECTOME TOPOLOGY ---
# We use a Scale-Free network to mimic the cortical "Rich Club"
# Node 0 is designated as the PCC/Precuneus Hub (MNI -2, -50, 25)
print(f"Mapping Connectome (N={N})...")
G = nx.barabasi_albert_graph(N, 3, seed=42)
adj_matrix = nx.to_numpy_array(G)

# --- 3. THE KURAMOTO EQUATION (PRR Implementation) ---
def kuramoto_dynamics(theta, t, omega, K, A):
    """
    dθ_i/dt = ω_i + (K/N) * Σ A_ij * sin(θ_j - θ_i)
    """
    # Calculate phase differences (matrix broadcasting)
    # theta_j - theta_i
    delta_theta = theta[None, :] - theta[:, None] 
    
    # Interaction term: Adjacency * sin(delta)
    interaction = A * np.sin(delta_theta)
    
    # Sum interaction for each node and scale by Coupling (K)
    dtheta = omega + (K / N) * np.sum(interaction, axis=1)
    return dtheta

# --- 4. INITIALIZATION ---
# Intrinsic Frequencies (Hz): 
# Hub = Gamma (40Hz), Periphery = Theta (6Hz) + Noise
omega = np.random.normal(loc=6.0, scale=1.0, size=N) 
omega[0] = 40.0  # The PCC Hub beats at Gamma

# Initial Phases (Random "Noise" State)
theta0 = np.random.uniform(0, 2*np.pi, N)

# Time vector
t = np.linspace(0, TIME_SEC, int(TIME_SEC/DT))

# --- 5. EXECUTION ---
print(f"Simulating PRR Dynamics (K={K_CRITICAL})...")
# Solve ODE
theta_t = odeint(kuramoto_dynamics, theta0, t, args=(omega, K_CRITICAL, adj_matrix))

# --- 6. ANALYTICS: Order Parameter (r) ---
# r(t) = |(1/N) * Σ e^(i*θ_t)|
order_param = np.abs(np.mean(np.exp(1j * theta_t), axis=1))

# --- 7. VISUALIZATION ---
plt.figure(figsize=(12, 5))

# Plot 1: Phase Synchronization over Time
plt.subplot(1, 2, 1)
plt.plot(t, order_param, color='gold', linewidth=2)
plt.axhline(y=0.9, color='r', linestyle='--', label='Zero-Lag Threshold')
plt.title(f"Simulated Wisdom State (K={K_CRITICAL})")
plt.xlabel("Time (s)")
plt.ylabel("Global Synchrony (r)")
plt.legend()

# Plot 2: Phase Offset of PCC Hub vs Frontal Node
# Ideally, PCC (Node 0) should LEAD Frontal (Node 10) by ~47ms
plt.subplot(1, 2, 2)
plt.plot(t[-500:], np.sin(theta_t[-500:, 0]), label='PCC Hub (Gamma)', color='gold')
plt.plot(t[-500:], np.sin(theta_t[-500:, 10]), label='Frontal Node', color='blue', alpha=0.6)
plt.title("Phase-Relay Offset (Zoom)")
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

print(f"Final Synchrony (r): {order_param[-1]:.4f}")
if order_param[-1] > 0.9:
    print(">> PREDICTION CONFIRMED: Zero-Lag State Achieved.")
else:
    print(">> FAILURE: Cognitive Rigidity Persists.")