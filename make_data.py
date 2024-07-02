import argparse
import os
import librosa
from augmentation import save_key_trans#, add_brown_noise, add_gaussian_noise, add_white_noise
from feature_extraction import extract_chromagram_stft, extract_mfcc, extract_spectrogram


def main(args):
    data_type = args.data_type

    sample_dir = "./sound_sample"
    sample_list = os.listdir(sample_dir)
    save_dir = f"./save_dir_{data_type}"
    os.makedirs(save_dir, exist_ok=True)

    for i in sample_list: 
        sample_name = os.path.splitext(i)[0]
        os.makedirs(f"./{save_dir}/{sample_name}", exist_ok=True)

    # data augmentation
    transpose_list = [-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
    for sample in sample_list:
        y, sr = librosa.load(sample_dir+"/"+sample, sr=None)
        length = len(sample)
        path = sample_dir+"/"+sample
        sample_name = sample[:length-4] #.mp3 빼기
        
        for i in transpose_list:
            for j in range(4):
                file, sr = save_key_trans(y, sr, f"{save_dir}/{sample_name}", sample_name, i,j, False)
                if data_type == "mfcc":
                    extract_mfcc(file, sr, f"{sample_name}_key_{i}_noise_{j}", f"{save_dir}/{sample_name}/", type="npy")
                elif data_type == "chro":
                    extract_chromagram_stft(file, sr, f"{sample_name}_key_{i}_noise_{j}", f"{save_dir}/{sample_name}/", type="npy")
                elif data_type == "spec":
                    extract_spectrogram(file, sr, f"{sample_name}_key_{i}_noise_{j}", f"{save_dir}/{sample_name}/", type="npy")
                
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default='mfcc')
    args = parser.parse_args()
    print(args)
    main(args)

