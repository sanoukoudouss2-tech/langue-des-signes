import numpy as np

def add_gaussian_noise(X, sigma=0.01):
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise

def time_shift(seq, max_shift=5):
    shift = np.random.randint(-max_shift, max_shift + 1)
    seq_shifted = np.roll(seq, shift, axis=0)
    if shift > 0:
        seq_shifted[:shift] = seq_shifted[shift]   # répète la première frame valide
    elif shift < 0:
        seq_shifted[shift:] = seq_shifted[shift - 1]
    return seq_shifted

def mirror_hands(seq):
    seq = seq.copy()
    hand1, hand2 = seq[:, :63].copy(), seq[:, 63:].copy()
    hand1[:, 0::3] *= -1   # inverse x (les x sont aux indices 0,3,6,... dans chaque bloc de 63)
    hand2[:, 0::3] *= -1
    seq[:, :63], seq[:, 63:] = hand2, hand1   # main gauche <-> main droite
    return seq

def augment_dataset(X, y, n_copies=2, sigma=0.01, max_shift=5, mirror=True):
    X_list, y_list = [X], [y]
    for _ in range(n_copies):
        X_noisy = np.array([time_shift(add_gaussian_noise(seq, sigma), max_shift) for seq in X])
        X_list.append(X_noisy)
        y_list.append(y)
    if mirror:
        X_mirrored = np.array([mirror_hands(seq) for seq in X])
        X_list.append(X_mirrored)
        y_list.append(y)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)
