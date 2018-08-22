import imageio
import numpy as np
from matplotlib import pyplot as plt
import cv2

live_measurement = True
lim_freq_min = 20               #BPM
lim_freq_max = 200              #BPM
fps = 30
Tmax = 15

if live_measurement:
    cap = cv2.VideoCapture(0)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./data/output.avi', fourcc, fps, (640, 480))
    num_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            # write the flipped frame
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if num_frames > Tmax*fps:
                break
        else:
            break

        num_frames = num_frames + 1

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    filename = './data/output.avi'
else:
    filename = './data/heartbeat_da.avi'

reader = imageio.get_reader(filename)
fps = reader.get_meta_data()['fps']
print(fps)
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
f.suptitle('Heart Beat Color Channels - Time Domain')
axarr[0].plot(time, mean_r, '-r'), axarr[0].grid()
axarr[1].plot(time, mean_g, '-g'), axarr[1].grid()
axarr[2].plot(time, mean_b, '-b'), axarr[2].grid()


freq = np.linspace(0.0, fps, n_frames)
half_freq = freq[:n_frames//2]
fft_r = np.fft.fft(mean_r-np.mean(mean_r)); power_spec_r = 2.0/n_frames * np.abs(fft_r[:n_frames//2])
fft_g = np.fft.fft(mean_g-np.mean(mean_g)); power_spec_g = 2.0/n_frames * np.abs(fft_g[:n_frames//2])
fft_b = np.fft.fft(mean_b-np.mean(mean_b)); power_spec_b = 2.0/n_frames * np.abs(fft_b[:n_frames//2])

f, axarr = plt.subplots(3, sharex=True, sharey=False)
f.suptitle('Heart Beat Color Channels - FFT')


indx_freq_min = round(lim_freq_min/60/fps*n_frames)
indx_freq_max = round(lim_freq_max/60/fps*n_frames)


max_freq = freq[np.argmax(power_spec_r[indx_freq_min:indx_freq_max])] + lim_freq_min/60
max_pwr = np.amax(power_spec_r[indx_freq_min:indx_freq_max])
axarr[0].plot(half_freq*60, power_spec_r, '-r'), axarr[0].grid()
axarr[0].text(max_freq*1.1*60, max_pwr*0.9, str(round(max_freq*60,2))+'bpm')

max_freq = freq[np.argmax(power_spec_g[indx_freq_min:indx_freq_max])] + lim_freq_min/60
max_pwr = np.amax(power_spec_g[indx_freq_min:indx_freq_max])
axarr[1].plot(half_freq*60, power_spec_g, '-g'), axarr[1].grid()
axarr[1].text(max_freq*1.1*60, max_pwr*0.9, str(round(max_freq*60,2))+'bpm')

max_freq = freq[np.argmax(power_spec_b[indx_freq_min:indx_freq_max])] + lim_freq_min/60
max_pwr = np.amax(power_spec_b[indx_freq_min:indx_freq_max])
axarr[2].plot(half_freq*60, power_spec_b, '-b'), axarr[2].grid()
axarr[2].text(max_freq*1.1*60, max_pwr*0.9, str(round(max_freq*60,2))+'bpm')

plt.show()





