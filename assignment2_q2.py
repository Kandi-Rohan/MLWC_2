import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def solve_q2():
    print("Running Question 2 Solution...")
    # Keep this seed fixed so every run produces the same comparison dataset.
    np.random.seed(67)

    # Build the square 16-QAM constellation from its four I and Q levels.
    coords = np.array([-3, -1, 1, 3])
    I, Q = np.meshgrid(coords, coords)
    symbols = I.flatten() + 1j * Q.flatten()

    # Normalize the constellation to unit average symbol energy.
    scale_factor = np.sqrt(10)
    symbols_norm = symbols / scale_factor

    snr_values = [0, 5, 10, 15, 20, 25, 30]
    n_samples_per_point = 200

    data = []

    print("Generating QAM dataset...")
    for snr_db in snr_values:
        # With unit symbol energy, N0 is the inverse linear SNR. The factor
        # of two distributes the complex noise power between I and Q.
        snr_lin = 10**(snr_db / 10)
        N_0 = 1.0 / snr_lin
        noise_std = np.sqrt(N_0 / 2)
        
        for symbol_id, symbol in enumerate(symbols_norm):
            noise_I = np.random.randn(n_samples_per_point) * noise_std
            noise_Q = np.random.randn(n_samples_per_point) * noise_std
            
            received_i = symbol.real + noise_I
            received_q = symbol.imag + noise_Q
            
            for k in range(n_samples_per_point):
                data.append({
                    'snr_db': snr_db,
                    'symbol_id': symbol_id,
                    'tx_I': symbol.real,
                    'tx_Q': symbol.imag,
                    'rx_I': received_i[k],
                    'rx_Q': received_q[k]
                })

    df2 = pd.DataFrame(data)

    # --- Part (a): Extract Cartesian and polar features ---
    df2['r'] = np.sqrt(df2['rx_I']**2 + df2['rx_Q']**2)
    # Use the phase expression specified in the assignment.
    df2['theta'] = np.arctan(df2['rx_Q'] / df2['rx_I'])

    # --- Part (b): Select K and compare feature representations ---
    print("Running K-means for Part (b)...")
    df2_25 = df2[df2['snr_db'] == 25].copy()
    X_cart_25 = df2_25[['rx_I', 'rx_Q']].values

    K_range = range(2, 21)
    inertias = []
    silhouettes = []

    for k in K_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = km.fit_predict(X_cart_25)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_cart_25, labels))

    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(K_range, inertias, marker='o')
    axes[0].set_xlabel('K')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Inertia vs K')
    axes[0].set_xticks(K_range)

    axes[1].plot(K_range, silhouettes, marker='o', color='orange')
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('Silhouette Coefficient')
    axes[1].set_title('Silhouette Coefficient vs K')
    axes[1].set_xticks(K_range)
    plt.tight_layout()
    plt.savefig('q2_part_b_elbow.png')
    print("Saved q2_part_b_elbow.png")

    # K=16 Plot
    km16 = KMeans(n_clusters=16, init='k-means++', n_init=10, random_state=42)
    labels_16 = km16.fit_predict(X_cart_25)

    plt.figure(figsize=(7, 6))
    plt.scatter(df2_25['rx_I'], df2_25['rx_Q'], c=labels_16, cmap='tab20', alpha=0.6, s=10)
    plt.scatter(km16.cluster_centers_[:, 0], km16.cluster_centers_[:, 1], c='red', marker='x', s=100, linewidths=3)
    plt.title('K-means Clusters at 25 dB (Cartesian)')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.grid(True)
    plt.savefig('q2_part_b_scatter.png')
    print("Saved q2_part_b_scatter.png")

    # Cluster labels are arbitrary, so purity maps each cluster to its most
    # common true symbol before counting correctly assigned samples.
    def cluster_purity(y_true, y_pred):
        purity = 0
        for cluster in np.unique(y_pred):
            idx = (y_pred == cluster)
            true_labels = y_true[idx]
            most_freq = mode(true_labels, keepdims=True).mode[0]
            purity += np.sum(true_labels == most_freq)
        return purity / len(y_true)

    y_true_25 = df2_25['symbol_id'].values

    # Set 1
    purity_1 = cluster_purity(y_true_25, labels_16)

    # Set 2
    X_polar_25 = df2_25[['r', 'theta']].values
    km_polar = KMeans(n_clusters=16, init='k-means++', n_init=10, random_state=42)
    labels_polar = km_polar.fit_predict(X_polar_25)
    purity_2 = cluster_purity(y_true_25, labels_polar)

    # Set 3
    X_all_25 = df2_25[['rx_I', 'rx_Q', 'r', 'theta']].values
    scaler = StandardScaler()
    X_all_25_sc = scaler.fit_transform(X_all_25)
    km_all = KMeans(n_clusters=16, init='k-means++', n_init=10, random_state=42)
    labels_all = km_all.fit_predict(X_all_25_sc)
    purity_3 = cluster_purity(y_true_25, labels_all)

    print("\nCluster Purity Results at 25 dB (K=16):")
    print(f"Feature Set 1 (Cartesian): {purity_1:.4f}")
    print(f"Feature Set 2 (Polar)    : {purity_2:.4f}")
    print(f"Feature Set 3 (Combined) : {purity_3:.4f}\n")

    # --- Part (c): Compare adaptive K-means with fixed learned centroids ---
    print("Evaluating strategies for Part (c)...")
    purity_adaptive = []
    purity_fixed = []

    # Fit the fixed model only at 25 dB, then reuse its learned centroids at
    # every SNR as required by the assignment.
    reference_data = df2[df2['snr_db'] == 25][['rx_I', 'rx_Q']].values
    fixed_kmeans = KMeans(
        n_clusters=16,
        init='k-means++',
        n_init=10,
        random_state=42,
    )
    fixed_kmeans.fit(reference_data)

    for snr in snr_values:
        df_snr = df2[df2['snr_db'] == snr]
        X_snr = df_snr[['rx_I', 'rx_Q']].values
        y_true_snr = df_snr['symbol_id'].values
        
        # Strategy 1: Adaptive
        km_ad = KMeans(n_clusters=16, init='k-means++', n_init=10, random_state=42)
        labels_ad = km_ad.fit_predict(X_snr)
        purity_adaptive.append(cluster_purity(y_true_snr, labels_ad))
        
        # Strategy 2: Fixed centroids learned from the 25 dB data.
        labels_fix = fixed_kmeans.predict(X_snr)
        purity_fixed.append(cluster_purity(y_true_snr, labels_fix))

    plt.figure(figsize=(8, 5))
    plt.plot(snr_values, purity_adaptive, marker='o', label='Adaptive K-Means')
    plt.plot(snr_values, purity_fixed, marker='s', label='Fixed Template')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Cluster Purity')
    plt.title('Part (c): Demodulation Strategies Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('q2_part_c_purity.png')
    print("Saved q2_part_c_purity.png")
    print("Question 2 complete.\n")

if __name__ == "__main__":
    solve_q2()
