import urllib.request
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Pobieranie obrazu
img_url = "https://upload.wikimedia.org/wikipedia/commons/8/88/Bright_red_tomato_and_cross_section02.jpg"
req = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.request.urlopen(req)
image_data = np.asarray(bytearray(resp.read()), dtype="uint8")
img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Wyświetlenie obrazu
plt.figure()
plt.imshow(img)
plt.title("oryginalne zdjecie")
plt.axis('off')
plt.show()

# Zmiana rozdzielczosci o 33,3%
h, w = img.shape[:2]
img_resized = cv2.resize(img, (w//3, h//3))

# Skala szarosci/grayscale
gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

# Obrot obrazu o 90 stopni
rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

# Obraz wynikowy
plt.figure()
plt.imshow(rotated, cmap='gray')
plt.title("wynik")
plt.axis('off')
plt.show()

# Obraz jako macierz
print("macierz obrazu:")
print(rotated)
