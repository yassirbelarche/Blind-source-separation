# # from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.image as img
# import numpy as np

# from skimage import io
# img = io.imread('image.png', as_gray=True)

# # def gray(image):
#     # img = Image.open(image, "rp").convert('LA')
#     # img.save('greyscale.png')

# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# img = img.imread('image.png')     
# gray = rgb2gray(img)    
# plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
# plt.show()

# # gray('image/violon_original.jpg')

# # image2 = img.imread('image/greyscale.png')
# # plt.title('image source 2')
# # plt.imshow(image2, cmap='gray')
# # plt.show()


# from skimage import color
# from skimage import io

# img = io.imread('test.jpg')
# imgGray = color.rgb2gray(img)

import cv2 as cv
img = cv.imread('violon_original.jpg', 0)
