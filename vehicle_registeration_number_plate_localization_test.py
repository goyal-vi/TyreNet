#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skimage import util, io, filters, data, exposure, color, morphology
import matplotlib.pyplot as plt
import numpy as np
#!pip install sklearn
import sklearn


# # Reading and displaying the image

# In[3]:


car_rgb = io.imread("images/01.jpg")
plt.imshow(car_rgb)


# # Grayscale conversion
# `From the input RGB image it has to be convert to gray scale and the 8-bit gray value is intended. `

# In[9]:





# In[201]:





# In[202]:


car_byte


# In[4]:


car_gray = color.rgb2gray(car_rgb)
plt.imshow(car_gray, cmap = plt.cm.gray)


# In[5]:


car_byte = util.img_as_ubyte(car_gray)
plt.imshow(car_byte, cmap = plt.cm.gray)


# In[6]:


car_byte


# In[79]:


high_contrast_car = exposure.equalize_hist(car_byte)
plt.imshow(high_contrast_car, cmap = plt.cm.gray)


# # Noise Reduction
# `We used median filtering method to reduce the paper and salt noise. We have used 3x 3 masks to get eight
# neighbors of a pixel and their consistent gray value. `

# In[7]:


med = filters.median(car_gray, morphology.disk(5))


# In[8]:


plt.imshow(med, cmap = plt.cm.gray)


# In[9]:


med


# # Contrast Enhancement
# `Using histogram equalization method the difference of each image is being enhanced. The function used to
# improvementthat is J=histeq(k); histeq enhances the contrast of the images by converting the values in an intensity
# image. When image pixel intensity of 8-neibourgh connectivity, we supply a preferred histogram, histeq chooses the
# grayscale conversion T to minimize
# │c1 (T (k))-c0 (k) │
# In below we state the change of histogram from original image and after smearing the contrast enhancement using
# histogram equalization. `

# In[10]:


#ok
img_eq = exposure.equalize_hist(med)
plt.imshow(img_eq, cmap = plt.cm.gray)


# In[11]:


#ok
byte_eq = util.img_as_ubyte(img_eq)


# In[173]:


# img_eq = exposure.equalize_hist(car_gray)
# plt.imshow(img_eq, cmap = plt.cm.gray)


# In[12]:


#ok
#in byte, after filter for noise reduction is applied
img_eq_exp = exposure.rescale_intensity(byte_eq, in_range=(200, 255))
plt.imshow(img_eq_exp, cmap = plt.cm.gray)


# In[149]:


# img_adeq = exposure.equalize_adapthist(car_gray, clip_limit=0.03)
# plt.imshow(img_adeq, cmap = plt.cm.gray)


# In[197]:


# img_adeq_exp = exposure.rescale_intensity(img_eq, in_range=(0.98, 200))
# plt.imshow(img_adeq_exp, cmap = plt.cm.gray)


# # Plate localization 
# ``

# In[13]:


car_edges = filters.sobel(img_eq_exp)
plt.imshow(car_edges, cmap = plt.cm.gray)


# In[228]:


# car_edges = filters.sobel(img_eq)
# plt.imshow(car_edges, cmap = plt.cm.gray)  #via histogram equalizer


# In[152]:


# image = exposure.rescale_intensity(car_edges, in_range=(0, 1))
# plt.imshow(image, cmap = plt.cm.gray)
# #histogram equilizer


# In[153]:


# img_eq = exposure.rescale_intensity(img_eq, in_range=(0., 1))
# plt.imshow(img_eq, cmap = plt.cm.gray)


# In[14]:


car_edges = util.img_as_ubyte(car_edges)


# In[80]:


image = exposure.rescale_intensity(car_edges, in_range= (30, 255))
plt.imshow(image, cmap = plt.cm.gray)


# In[81]:


image


# In[82]:


import numpy as np
from skimage.morphology import reconstruction

seed = np.copy(image)
seed[1:-1, 1:-1] = image.max()
mask = image

filled = reconstruction(seed, mask, method='erosion')


# In[83]:


eroded = morphology.binary_erosion(image)


# In[84]:


plt.imshow(eroded, cmap = plt.cm.gray)


# In[85]:


plt.imshow(filled, cmap = plt.cm.gray)


# In[86]:


filled = filled.astype(int)


# In[87]:


cleaned = morphology.remove_small_objects(filled, min_size=1000, connectivity=4)


# In[88]:


plt.imshow(cleaned, cmap = plt.cm.gray)


# In[ ]:





# In[89]:


final = morphology.binary_erosion(cleaned)
plt.imshow(final , cmap = plt.cm.gray)


# In[ ]:





# In[25]:


np.shape(final)


# In[26]:


# med_ = filters.median(final, morphology.disk(2))
# plt.imshow(med_, cmap = plt.cm.gray)


# In[27]:


# np.ones(3)


# In[92]:


final_ = filters.rank.maximum(final,np.ones((25,25), dtype = int))
plt.imshow(final_, cmap = plt.cm.gray)


# In[97]:


val = (final_ > 0)*1
val_ = val*car_gray
plt.imshow(val_, cmap = plt.cm.gray)
io.imsave('images/temp.jpg', val_)
