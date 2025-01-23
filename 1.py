import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

image_path = r"T3.jpg"

image = cv2.imread(image_path)

if image is None:
    print("tasvir yaft nashod. lotfan masir digari entekhab konid.")
else:
    print("name file:", image_path.split("\\")[-1]) 
    print("format file:", image_path.split(".")[-1]) 
    print("abaad tasvir:", image.shape) 
    print("noe dade tasvir:", image.dtype)
    print("omghe range tasvir:", image.shape[2] if len(image.shape) > 2 else 1) 

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title("tasvir asli")
    plt.axis('off')
    plt.show()

    b_channel, g_channel, r_channel = cv2.split(image)

    plt.imshow(r_channel, cmap='Reds')
    plt.title("kanal ghermez : (R)")
    plt.axis('off')
    plt.show()

    plt.imshow(g_channel, cmap='Greens')
    plt.title("kanal sabz : (G)")
    plt.axis('off')
    plt.show()

    plt.imshow(b_channel, cmap='Blues')
    plt.title("kanal abi : (B)")
    plt.axis('off')
    plt.show()

    resized_image = cv2.resize(image_rgb, (0, 0), fx=0.5, fy=0.5)
    plt.imshow(resized_image)
    plt.title("tasvir (50%) kochak shodeh")
    plt.axis('off')
    plt.show()

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_image, cmap='gray')
    plt.title("tasvir khakestari")
    plt.axis('off')
    plt.show()

    pixels = image_rgb.reshape((-1, 3))
    kmeans = KMeans(n_clusters=16, random_state=42)
    kmeans.fit(pixels)
    indexed_pixels = kmeans.cluster_centers_[kmeans.labels_].astype('uint8')
    indexed_image = indexed_pixels.reshape(image_rgb.shape)

    plt.imshow(indexed_image)
    plt.title("tasvir index shode ba 16 rang")
    plt.axis('off')
    plt.show()

    unique_colors = np.unique(indexed_pixels, axis=0)
    plt.figure(figsize=(8, 2))
    plt.imshow([unique_colors], aspect='auto')
    plt.title("jadval rang 16 taei")
    plt.axis('off')
    plt.show()

    complement_image = 255 - image_rgb
    plt.imshow(complement_image)
    plt.title("tasvir mokamel rangi")
    plt.axis('off')
    plt.show()

    sum_image = cv2.add(image_rgb, complement_image)  
    diff_image = cv2.subtract(image_rgb, complement_image)  

    plt.imshow(sum_image)
    plt.title("majmoe tasvir asli va mokamel")
    plt.axis('off')
    plt.show()

    plt.imshow(diff_image)
    plt.title("tafazole tasvir asli va mokamel")
    plt.axis('off')
    plt.show()
