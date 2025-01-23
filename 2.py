import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r"T4.png"

image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image_gray is None:
    print("tasvir yaft nashod. lotfan masir digari entekhab konid.")
else:
    plt.imshow(image_gray, cmap='gray')
    plt.title("tasvir khakestari")
    plt.axis('off')
    plt.show()

    plt.hist(image_gray.ravel(), bins=256, range=[0, 256], color='black')
    plt.title("histogeram tasvir")
    plt.xlabel("maghadir pixel")
    plt.ylabel("tedad")
    plt.show()

    equalized_image = cv2.equalizeHist(image_gray)
    plt.imshow(equalized_image, cmap='gray')
    plt.title("tasvir ba taadil histogaram")
    plt.axis('off')
    plt.show()

    plt.hist(equalized_image.ravel(), bins=256, range=[0, 256], color='black')
    plt.title("histogeram tasvir taadil dade shode")
    plt.xlabel("maghadir pixel")
    plt.ylabel("tedad")
    plt.show()

    _, binary_image = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)
    plt.imshow(binary_image, cmap='gray')
    plt.title("tasvir bainary shode")
    plt.axis('off')
    plt.show()

    low_pass_image = cv2.GaussianBlur(image_gray, (5, 5), 0)
    plt.imshow(low_pass_image, cmap='gray')
    plt.title("tasvir baad az filter paeein gozar")
    plt.axis('off')
    plt.show()

    high_pass_image = cv2.Laplacian(image_gray, cv2.CV_64F)
    high_pass_image = np.uint8(np.absolute(high_pass_image))
    plt.imshow(high_pass_image, cmap='gray')
    plt.title("tasvir baad az filter bala gozar")
    plt.axis('off')
    plt.show()

    canny_edges = cv2.Canny(image_gray, 100, 200)
    plt.imshow(canny_edges, cmap='gray')
    plt.title("labe yabi ba filter canny")
    plt.axis('off')
    plt.show()

    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    sobel_edges = np.uint8(sobel_edges)
    plt.imshow(sobel_edges, cmap='gray')
    plt.title("labe yabi ba filter sobel")
    plt.axis('off')
    plt.show()
