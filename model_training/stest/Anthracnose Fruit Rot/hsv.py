import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def augment_hsv(image, h_gain=0.2, s_gain=0.7, v_gain=0.4):
    """
    Menerapkan augmentasi HSV pada gambar.

    Parameter:
      - image: Gambar input dalam format BGR.
      - h_gain: Rentang perubahan hue (±0.015).
      - s_gain: Rentang perubahan saturasi ([1-s_gain, 1+s_gain]).
      - v_gain: Rentang perubahan value ([1-v_gain, 1+v_gain]).

    Mengembalikan:
      - img_aug: Gambar yang telah diaugmentasi.
      - (delta_h, factor_s, factor_v): Tuple berisi nilai perubahan HSV.
    """
    # Konversi gambar dari BGR ke HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    
    # Menentukan perubahan secara acak
    delta_h = random.uniform(-h_gain, h_gain) * 180  # OpenCV menggunakan rentang 0-179 untuk hue
    factor_s = random.uniform(1 - s_gain, 1 + s_gain)
    factor_v = random.uniform(1 - v_gain, 1 + v_gain)
    
    # Terapkan perubahan
    h = (h.astype(np.float32) + delta_h) % 180  # pastikan hue tetap dalam [0, 179]
    s = s.astype(np.float32) * factor_s
    v = v.astype(np.float32) * factor_v
    
    # Batasi nilai saturasi dan value agar tidak melebihi 255
    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)
    
    hsv_aug = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)])
    img_aug = cv2.cvtColor(hsv_aug, cv2.COLOR_HSV2BGR)
    
    return img_aug, (delta_h, factor_s, factor_v)

if __name__ == '__main__':
    # Input path gambar
    img_path = input("Masukkan path gambar (misal: gambar.jpg): ")
    image = cv2.imread(img_path)
    if image is None:
        print("Gambar tidak ditemukan, periksa path yang diberikan.")
        exit(1)
    
    # Input jumlah gambar augmentasi yang diinginkan
    try:
        num_aug = int(input("Masukkan jumlah gambar augmentasi yang diinginkan: "))
    except ValueError:
        print("Masukkan angka yang valid.")
        exit(1)
    
    # Menentukan layout grid (misal: 3 gambar per baris)
    cols = 3
    rows = (num_aug + cols - 1) // cols  # menghitung jumlah baris yang diperlukan
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    # Pastikan axes berupa list (flatten) jika lebih dari 1 subplot
    if num_aug == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(num_aug):
        aug_img, (delta_h, factor_s, factor_v) = augment_hsv(image)
        # Konversi gambar dari BGR ke RGB untuk matplotlib
        aug_img_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(aug_img_rgb)
        axes[i].axis('off')
        # Ganti judul (title) dengan nilai HSV
        title = f"Hue: {delta_h:.2f}, Sat: {factor_s:.2f}, Val: {factor_v:.2f}"
        axes[i].set_title(title, fontsize=12)
    
    # Sembunyikan subplot yang tidak terpakai
    for j in range(num_aug, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
