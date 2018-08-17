import pylab
import imageio
import numpy as np
from matplotlib import pyplot as plt

filename = './data/heartbeat_da.avi'
vid = imageio.get_reader(filename)
n_frames = vid.get_length()
mean_r = np.zeros(n_frames)
mean_g = np.zeros(n_frames)
mean_b = np.zeros(n_frames)
time_indx = np.arange(n_frames)

indx = 0
for img in vid:
    np_img = np.array(img)
    mean_r[indx] = np.mean(img[:, :, 0])
    mean_g[indx] = np.mean(img[:, :, 1])
    mean_b[indx] = np.mean(img[:, :, 2])
    indx = indx + 1

fig_red = plt.figure(),  plt.plot(time_indx, mean_r, '-r')
fig_gre = plt.figure(),  plt.plot(time_indx, mean_g, '-g')
fig_blu = plt.figure(),  plt.plot(time_indx, mean_b, '-b')
plt.show()





