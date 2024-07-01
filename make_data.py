import os
import librosa
from augmentation import save_key_trans, add_brown_noise, add_gaussian_noise, add_white_noise
from feature_extraction import extract_chromagram_stft, extract_mfcc, extract_spectrogram

# generate save folder
sample_dir = "./sound_sample"
sample_list = os.listdir(sample_dir)
print(sample_list)
#sample_name_list =[]
save_dir = "./save_dir_mfcc"
os.makedirs(save_dir, exist_ok=True)
for i in sample_list: 
    sample_name = os.path.splitext(i)[0]
    #sample_name_list.append(sample_name)
    os.makedirs(f"./{save_dir}/{sample_name}", exist_ok=True)

# data augmentation
transpose_list = [-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
for sample in sample_list:
    y, sr = librosa.load(sample_dir+"/"+sample, sr=None)
    length = len(sample)
    path = sample_dir+"/"+sample
    sample_name = sample[:length-4] #.mp3 빼기
    # print(sample_name)
    
    for i in transpose_list:
        for j in range(4):
            file, sr = save_key_trans(y, sr, f"{save_dir}/{sample_name}", sample_name, i,j, False)
            extract_mfcc(file, sr, f"sample_name_key_{i}_noise_{j}", f"{save_dir}/{sample_name}/", type="npy")
            #extract_mfcc(file, sr, f"sample_name_key_{i}_noise_{j}", f"{save_dir}/{sample_name}/", type="jpg")
        


