import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def solve_q1():
    print("Running Question 1 Solution...")
    # Use the supplied dataset so that the channel parameters and random seed
    # remain exactly those specified for the assignment.
    try:
        df = pd.read_csv('los_nlos_dataset.csv')
    except FileNotFoundError:
        print("los_nlos_dataset.csv not found. Attempting to run generate_dataset.py...")
        import generate_dataset
        df = generate_dataset.build_and_save_csv('los_nlos_dataset.csv')

    number_of_taps = 6

    def extract_features(row):
        """Calculate the five channel features required by Question 1."""
        channel_coefficients = np.array([
            row[f'h_real_{tap}'] + 1j * row[f'h_imag_{tap}']
            for tap in range(number_of_taps)
        ])
        tap_delays = np.array([
            row[f'tau_{tap}'] for tap in range(number_of_taps)
        ])
        
        tap_amplitudes = np.abs(channel_coefficients)
        tap_powers = tap_amplitudes**2
        
        # Kurtosis and skewness describe the shape of the received-power
        # distribution across the multipath taps.
        mean_power = np.mean(tap_powers)
        power_standard_deviation = np.std(tap_powers, ddof=0)
        
        if power_standard_deviation > 0:
            kurtosis = np.mean((tap_powers - mean_power)**4) / (power_standard_deviation**4)
        else:
            kurtosis = 0.0
            
        if power_standard_deviation > 0:
            skewness = np.mean((tap_powers - mean_power)**3) / (power_standard_deviation**3)
        else:
            skewness = 0.0
            
        # Rising time is the delay of the strongest tap relative to the first tap.
        rising_time = tap_delays[np.argmax(tap_amplitudes)] - np.min(tap_delays)
        
        # RMS delay spread is the power-weighted standard deviation of delays.
        total_power = np.sum(tap_powers)
        if total_power > 0:
            mean_delay = np.sum(tap_delays * tap_powers) / total_power
            rms_delay_spread = np.sqrt(
                np.sum(((tap_delays - mean_delay)**2) * tap_powers) / total_power
            )
        else:
            rms_delay_spread = 0.0
            
        # Estimate the K-factor from the peak received power relative to the
        # remaining multipath power, matching the supplied generator's check.
        peak_power = np.max(tap_powers)
        remaining_power = total_power - peak_power
        if remaining_power > 0:
            k_factor = 10 * np.log10(peak_power / remaining_power)
        else:
            k_factor = 0.0
        
        return pd.Series([kurtosis, skewness, rising_time, rms_delay_spread, k_factor])

    print("Extracting features (this might take a few seconds)...")
    df[['kurtosis', 'skewness', 'rising_time', 'rms_delay_spread', 'k_factor']] = df.apply(extract_features, axis=1)

    # --- Part (b) ---
    snr_values = np.sort(df['snr_db'].unique())

    feature_sets = {
        1: ['kurtosis'],
        2: ['skewness'],
        3: ['rising_time'],
        4: ['rms_delay_spread'],
        5: ['k_factor'],
        6: ['kurtosis', 'skewness', 'rising_time', 'rms_delay_spread', 'k_factor']
    }

    accuracies = {i: [] for i in range(1, 7)}
    models_25db = {}
    scalers_25db = {}

    print("Training models for Part (b)...")
    for snr in snr_values:
        df_snr = df[df['snr_db'] == snr]
        
        for i in range(1, 7):
            X = df_snr[feature_sets[i]]
            y = df_snr['label']
            
            # 80/20 train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)
            
            clf = SVC(kernel='rbf', C=1.0)
            clf.fit(X_train_sc, y_train)
            
            acc = clf.score(X_test_sc, y_test)
            accuracies[i].append(acc * 100)
            
            # Save the combined model for part (c)
            if snr == 25 and i == 6:
                models_25db[6] = clf
                scalers_25db[6] = scaler

    plt.figure(figsize=(10, 6))
    feature_labels = {
        1: 'SVM-1 (Kurtosis)',
        2: 'SVM-2 (Skewness)',
        3: 'SVM-3 (Rising time)',
        4: 'SVM-4 (RMS delay spread)',
        5: 'SVM-5 (Rician K-factor)',
        6: 'SVM-6 (All five features)',
    }
    for feature_set_number in range(1, 7):
        plt.plot(
            snr_values,
            accuracies[feature_set_number],
            marker='o',
            label=feature_labels[feature_set_number],
        )

    plt.xlabel('SNR (dB)')
    plt.ylabel('Classification Accuracy (%)')
    plt.title('Part (b): Classification Accuracy vs. SNR')
    plt.legend()
    plt.grid(True)
    plt.savefig('q1_part_b_accuracy.png')
    print("Saved q1_part_b_accuracy.png")

    # --- Part (c) ---
    print("Evaluating for Part (c)...")
    fixed_clf = models_25db[6]
    fixed_scaler = scalers_25db[6]

    accuracies_fixed = []
    for snr in snr_values:
        df_snr = df[df['snr_db'] == snr]
        X = df_snr[feature_sets[6]]
        y = df_snr['label']
        
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_test_sc = fixed_scaler.transform(X_test)
        acc = fixed_clf.score(X_test_sc, y_test)
        accuracies_fixed.append(acc * 100)

    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, accuracies[6], marker='o', label='Train = Test SNR')
    plt.plot(snr_values, accuracies_fixed, marker='s', label='Train at 25 dB')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Classification Accuracy (%)')
    plt.title('Part (c): Adaptive vs Fixed SNR Training')
    plt.legend()
    plt.grid(True)
    plt.savefig('q1_part_c_accuracy.png')
    print("Saved q1_part_c_accuracy.png")
    print("Question 1 complete.\n")

if __name__ == "__main__":
    solve_q1()
