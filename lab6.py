# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:08:16 2024

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:56:22 2024

@author: User
"""

#import sys
#sys.path.append('../')
import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import data
from scipy import ndimage
import matplotlib.pyplot as plt
#from utility import segmentation_utils

def hue_eq(image):
    
    channels = [0]
    histSize = [180]
    qqq = [0, 180]
    
    hist1 = cv.calcHist([image], channels, None, histSize, qqq)
    
    lut = np.zeros([180, 1]) 
    
    hsum = hist1.sum()
    for i in range(180):
        lut[i] = np.uint8(179 * hist1[:i].sum()/hsum)
        
    image2 = image.copy()
        
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image2[i][j] = lut[image2[i][j]]
            
    return image2

def eq(image):
    
    channels = [0]
    histSize = [256]
    qqq = [0, 256]
    
    hist1 = cv.calcHist([image], channels, None, histSize, qqq)
    
    lut = np.zeros([256, 1]) 
    
    hsum = hist1.sum()
    for i in range(256):
        lut[i] = np.uint8(255 * hist1[:i].sum()/hsum)
        
    image2 = image.copy()
        
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image2[i][j] = lut[image2[i][j]]
            
    return image2




image = cv.imread('./pants.jpg')
image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

h, s, v = cv.split(image_hsv)
h = hue_eq(h)
#s = eq(s)
#v = eq(v)
image_hsv = cv.merge([h, s, v])
image_to_clusterization = cv.cvtColor(image_hsv, cv.COLOR_HSV2RGB)
## Методы кластеризации. Сдвиг среднего (Mean shift)
# Сглаживаем чтобы уменьшить шум
blur_image = cv.medianBlur(image_to_clusterization, 11)
# Выстраиваем пиксели в один ряд и переводим в формат с правающей точкой
flat_image = np.float32(blur_image.reshape((-1,3)))

# Используем meanshift из библиотеки sklearn
bandwidth = estimate_bandwidth(flat_image, quantile=.16, n_samples=5000)
ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
ms.fit(flat_image)
labeled = ms.labels_

# получим количество сегментов
segments = np.unique(labeled)
print('Number of segments: ', segments.shape[0])

# получим средний цвет сегмента
total = np.zeros((segments.shape[0], 3), dtype=float)
count = np.zeros(total.shape, dtype=float)
for i, label in enumerate(labeled):
    total[label] = total[label] + flat_image[i]
    count[label] += 1
avg = total/count
avg = np.uint8(avg)
max_red = max(avg, key = lambda i: i[0] / (i[0]/3+i[1]/3+i[2]/3))[0] #находим цвет с наибольшей долей красного
# Для каждого пискеля проставим средний цвет его сегмента
mean_shift_image = avg[labeled].reshape((image.shape))
# Маской скроем один из сегментов
mask1 = mean_shift_image[:,:,0]
mask1[mask1!=max_red] = 0 #здесь фильтрация по цвету
# в исходном изображении нас интересовал кластер, получивший цвет
# с красным каналом = 89
#TODO подобрать цвет, соотв. штанам
mean_shift_with_mask_image = cv.bitwise_and(image, image, mask=mask1)
# Построим изображение
plt.figure(figsize=(15,20))
plt.subplot(1, 4, 1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.subplot(1, 4, 2)
plt.imshow(image_to_clusterization)
plt.subplot(1, 4, 3)
plt.imshow(mean_shift_image, cmap='Set3')
plt.subplot(1, 4, 4)
plt.imshow(cv.cvtColor(mean_shift_with_mask_image, cv.COLOR_BGR2RGB))
plt.show()