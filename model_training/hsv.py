import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def augment_hsv(image, h_gain=0.015, s_gain=0.7, v_gain=0.4):
    """
    Menerapkan augmentasi HSV pada gambar.
    
    Parameter:
      - image: Gambar input dalam format BGR.
      - h_gain: Rentang perubahan hue (±0.015).
      - s_gain: Rentang perubahan saturasi ([1-s_gain, 1+s_gain]).
      - v_gain: Rentang perubahan value/kecerahan ([1-v_gain, 1+v_gain]).
      
    Mengembalikan gambar yang telah diubah.
    """
    # Ubah gambar dari BGR ke HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    
    # Tentukan perubahan secara acak
    # Pada OpenCV, nilai hue berada dalam rentang [0, 179]
    delta_h = random.uniform(-h_gain, h_gain) * 180  
    factor_s = random.uniform(1 - s_gain, 1 + s_gain)
    factor_v = random.uniform(1 - v_gain, 1 + v_gain)
    
    # Terapkan perubahan
    h = (h.astype(np.float32) + delta_h) % 180  # Pastikan nilai hue berada di [0, 179]
    s = s.astype(np.float32) * factor_s
    v = v.astype(np.float32) * factor_v
    
    # Pastikan nilai saturasi dan value tidak melebihi 255
    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)
    
    hsv_aug = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)])
    img_aug = cv2.cvtColor(hsv_aug, cv2.COLOR_HSV2BGR)
    return img_aug

if __name__ == "__main__":
    # Input path gambar
    img_path = input("Masukkan path gambar (misal: gambar.jpg): ")
    image = cv2.imread(img_path)
    
    if image is None:
        print("Gambar tidak ditemukan. Pastikan path benar.")
    else:
        # Tampilkan gambar asli dan beberapa variasi augmentasi
        plt.figure(figsize=(12, 6))
        
        # Gambar asli
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis("off")
        
        # Tampilkan 3 gambar hasil augmentasi
        for i in range(3):
            img_aug = augment_hsv(image)
            plt.subplot(1, 4, i + 2)
            plt.imshow(cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB))
            plt.title(f"Augmented {i+1}")
            plt.axis("off")
            
        plt.tight_layout()
        plt.show()
