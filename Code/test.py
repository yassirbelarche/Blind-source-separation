import matplotlib.pyplot as plt
import matplotlib.image as img
# from PIL import Image


image = img.imread('justice.jpg')
# image = Image.open('test1_gray')
plt.imshow(image)
plt.show()


