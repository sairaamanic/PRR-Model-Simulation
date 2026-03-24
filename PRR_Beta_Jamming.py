import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# --- 1. Generate Scale-Free "Lichtman" Connectome ---
N = 100
# m=3 means each new node attaches to 3 existing nodes (creates hubs)
G = nx.barabasi_albert_graph(N, 3) 
adj = nx.to_numpy_array(G)

# --- 2. Simulation Parameters ---
dt = 0.01
T = 10.0
steps = int(T / dt)

# Frequencies: 18Hz Beta (Jamming) for periphery, 6Hz Theta for PCC (Node 0)
freqs = np.random.normal(18, 1, N)
freqs[0] = 6.0 

# Coupling Strength (The "Hardware" Integrity)
K = 8.0  

# Initialize Phases
phases = np.random.uniform(0, 2 * np.pi, N)
history = np.zeros((steps, N))
order_parameter = np.zeros(steps)

# --- 3. Run Simulation ---
for t in range(steps):
    # Calculate Phase-Amplitude Coupling / Synchrony (r)
    r = np.abs(np.mean(np.exp(1j * phases)))
    order_parameter[t] = r
    
    # Scale-Free Kuramoto Equation
    # Only connected nodes influence each other
    coupling = np.zeros(N)
    for i in range(N):
        # sum(K * adj_ij * sin(theta_j - theta_i))
        coupling[i] = (K / N) * np.sum(adj[i] * np.sin(phases - phases[i]))
    
    d_phases = 2 * np.pi * freqs + coupling
    phases += d_phases * dt
    history[t, :] = np.sin(phases)

# --- 4. Visualization ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
nx.draw(G, node_size=20, node_color='blue', alpha=0.5)
plt.title("Scale-Free Connectome (Hardware)")

plt.subplot(1, 2, 2)
plt.plot(np.linspace(0, T, steps), order_parameter, color='red')
plt.axhline(y=0.5, color='black', linestyle='--', label='Refractive Failure Threshold')
plt.title("Synchrony (r) over Time")
plt.xlabel("Time (s)")
plt.ylabel("Coherence")
plt.legend()

plt.tight_layout()
plt.show()