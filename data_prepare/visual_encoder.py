import numpy as np
import matplotlib.pyplot as plt
from typing import List

class VisualEncoder:
    def __init__(self):
        self.colors = {
            'I': [1.0, 0.0, 0.0], 'S': [1.0, 0.0, 1.0],
            'D': [0.0, 1.0, 0.0], 'M': [0.5, 0.5, 0.5],
            '=': [0.5, 0.5, 0.5], 'X': [0.5, 0.5, 0.5],
            '-': [0.85, 0.85, 0.85]
        }

    def save_cigar_image(self, matrix: List[List[str]], output_path: str):
        if not matrix: return
        h, w = len(matrix), len(matrix[0])
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                rgb[i, j] = self.colors.get(matrix[i][j], self.colors['-'])
        plt.imsave(output_path, rgb)