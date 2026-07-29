import numpy as np

def normalize_hand(pts):
    """pts : array de forme (63,) = 21 points x,y,z flatten"""
    pts = pts.reshape(21, 3)
    
    # 1. Invariance à la translation : le poignet devient l'origine (0,0,0)
    wrist = pts[0].copy()
    pts = pts - wrist
    
    # 2. Invariance à l'échelle : on divise par une distance de référence
    # (poignet -> base du majeur, landmark 9), stable quelle que soit la distance à la caméra
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:      # évite une division par zéro si la main est mal détectée
        scale = 1e-6
    pts = pts / scale
    
    return pts.flatten()