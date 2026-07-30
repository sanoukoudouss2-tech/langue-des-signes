import numpy as np


def add_gaussian_noise(X, sigma=0.01):
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise


def time_shift(seq, max_shift=5):
    shift = np.random.randint(-max_shift, max_shift + 1)
    seq_shifted = np.roll(seq, shift, axis=0)
    if shift > 0:
        seq_shifted[:shift] = seq_shifted[shift]  # répète la première frame valide
    elif shift < 0:
        seq_shifted[shift:] = seq_shifted[shift - 1]
    return seq_shifted


def mirror_hands(seq):
    """
    seq : (T, 132), bloc par main = 66 valeurs contiguës
          [0:63]  = forme normalisée (21 points x,y,z)
          [63:66] = trajectoire du poignet (x,y,z)
    main gauche = seq[:, :66], main droite = seq[:, 66:]
    """
    seq = seq.copy()
    hand1, hand2 = seq[:, :66].copy(), seq[:, 66:].copy()

    # inverse la composante x : indices 0,3,...,60 pour la forme, et index 63 pour la trajectoire
    for hand in (hand1, hand2):
        hand[:, 0:63:3] *= -1  # x de chaque landmark de la forme
        hand[:, 63] *= -1      # x de la trajectoire du poignet

    seq[:, :66], seq[:, 66:] = hand2, hand1  # main gauche <-> main droite
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