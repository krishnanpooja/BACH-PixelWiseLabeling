import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread("/home/deepita/code/xml/A01.png")
#plt.imshow(image)
fig=plt.figure()
plt.plot(range(10))

fig.savefig("/home/deepita/code/sample.png")
#plt.show()
