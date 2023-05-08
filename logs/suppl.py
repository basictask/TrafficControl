"""
Supplementary methods used across all the notebooks responsible for generating plots
"""
import os
import matplotlib.pyplot as plt

IMAGES_PATH = './plots'


def save_fig(fig_id: str, tight_layout: bool = True, fig_extension: str = "png", resolution: int = 300):
    path = os.path.join(IMAGES_PATH, fig_id + '.' + fig_extension)
    print('Saving figure', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
    
def letter_to_number(letter: str) -> int:
    if len(letter) == 1:  # A, B, C
        return ord(letter) - ord('A')
    else:  # A1, B1, C1
        return int(letter[1:]) * 26 + ord(letter[0]) - ord('A')

def number_to_letter(num: int) -> str:
    if num < 26:
        return chr(num + 65)
    else:
        return chr(num % 26 + 65) + str(num // 26)
    