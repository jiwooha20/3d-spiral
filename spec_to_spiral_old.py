import numpy as np
import librosa
import librosa.display

import matplotlib.pyplot as plt
import os
import math

# convert spectrogram to spiral
test = np.load("./save_dir/001_spec.npy")
test = 1 - (test+80)/80


time_slot = test.shape[1]
freq_slot = test.shape[0]

def polar_to_cartesian(r, theta):
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y

sampling_rate = 44100 # Hz
freq_initial = 55 # initial freq A1
r_init = 1
theta_init = (np.pi/2)

def cal_r(freq):

    if freq<55:
        return 14/15.
    elif freq>=55 and freq<110:
        return 13/15
    elif freq>=110 and freq<220:
        return 12/15
    elif freq>=220 and freq<440:
        return 11/15
    elif freq>=440 and freq<880:
        return 10/15
    elif freq>=880 and freq<1760:
        return 9/15
    elif freq>=1760 and freq<3520:
        return 8/15
    elif freq>=3520 and freq<7040:
        return 7/15
    elif freq>=7040 and freq<14080:
        return 6/15
    elif freq>=14080 and freq<28160:
        return 5/15
    else: return 0.

real_time = time_slot * (512/sampling_rate)


freq_list = np.array(range(1025)[1:])*(22050/1024)
print(freq_list)
r_list = []
for i in freq_list:
    r_list.append(cal_r(i))
theta_list = (np.pi/2) - 2*np.pi*np.log2(freq_list/freq_initial)

numpy_list = []
for i in range(time_slot):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    A = test[1:,i]
    for j in range(len(freq_list)):
        if A[j] == 1: continue
        else:
            colors = (A[j], A[j], A[j])  
            ax.scatter(theta_list[j],r_list[j], marker='o', s=1, c = colors)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.grid(False)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    numpy_list.append(gray_image)
    if i>4300 and i<5000:
        plt.imsave(f"./save_001_new/polar_coordinates_plot_{i}.jpg", gray_image, cmap='gray')
    plt.close(fig)
    print(i)

stacked_array = np.stack(numpy_list)
np.save(f"./save_001_new/polar_coordinates_plot.npy", stacked_array)
print(stacked_array.shape)