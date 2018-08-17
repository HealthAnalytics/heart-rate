import pylab
import imageio
import numpy as np
from matplotlib import pyplot as plt

filename = './data/heartbeat_da.avi'
vid = imageio.get_reader(filename)
n_frames = vid.get_length()
print(n_frames)
for img in vid:
    #imgplot = plt.imshow(img)
    pylab.imshow(img)
