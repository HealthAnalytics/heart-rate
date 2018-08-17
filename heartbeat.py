import imageio
import numpy as np
from matplotlib import pyplot as plt
plt.close('all')

filename = './data/heartbeat_da.avi'
reader = imageio.get_reader(filename)
fps = reader.get_meta_data()['fps']
n_frames = reader.get_length()
mean_r = np.zeros(n_frames)
mean_g = np.zeros(n_frames)
mean_b = np.zeros(n_frames)
time = np.arange(n_frames)*(1/fps)
sample_time = (1/fps)
indx = 0
for img in reader:
    mean_r[indx] = np.mean(img[:, :, 0])
    mean_g[indx] = np.mean(img[:, :, 1])
    mean_b[indx] = np.mean(img[:, :, 2])
    indx = indx + 1


f, axarr = plt.subplots(3, sharex=True, sharey=False)
f.suptitle('Heart Beat Color Channels')
axarr[0].plot(time, mean_r, '-r')
axarr[1].plot(time, mean_g, '-g')
axarr[2].plot(time, mean_b, '-b')


yf = np.fft.fft(mean_r-np.mean(mean_r))
xf = np.linspace(0.0, 1.0/(2.0*sample_time), n_frames/2)*60

fig, ax = plt.subplots()
ax.plot(xf[:n_frames//2], 2.0/n_frames * np.abs(yf[:n_frames//2]))
plt.show()





