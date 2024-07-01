import librosa
import librosa.display
import numpy as np
import soundfile as sf


def save_key_trans(y, sr, dirname, name, key_shift, noise, save=False):
    y_norm = y / np.max(np.abs(y)) # normalization
    if key_shift != 0:
        y_norm = librosa.effects.pitch_shift(y_norm, sr=sr, n_steps=key_shift)
    output_file=f"./{dirname}/{name}_KEY_{key_shift}.mp3"
    if save:
        sf.write(output_file, y_norm, sr)
    
    if (noise != 0):
        if noise == 1:
            y_noise = add_white_noise(y_norm,sr)
        elif noise == 2:
            y_noise = add_brown_noise(y_norm,sr)
        elif noise ==3:
            y_noise = add_gaussian_noise(y_norm,sr)
    
        with_noise_file_path = f"./{dirname}/{name}_KEY_{key_shift}_with_noise_{noise}.mp3"

        if save:
            sf.write(with_noise_file_path, y_noise, sr)
    
        return y_norm, sr 
    return y_norm, sr


def add_white_noise(y, sr, noise_level=0.05):
    white_noise = np.random.randn(len(y))
    augmented_data = y + noise_level * white_noise
    return augmented_data


def add_brown_noise(y, sr, noise_level=0.05):
    brown_noise = np.cumsum(np.random.randn(len(y)))
    brown_noise = brown_noise / np.max(np.abs(brown_noise))  # Normalize brown noise
    augmented_data = y + noise_level * brown_noise
    return augmented_data


def add_gaussian_noise(y, sr, noise_level=0.05):
    gaussian_noise = np.random.normal(0, 1, len(y))
    augmented_data = y + noise_level * gaussian_noise
    return augmented_data 
    

if __name__ == "__main__":
    transpose_list = [-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
    for i in transpose_list:
        for j in range(4):
            save_key_trans("./sound_sample/001.mp3", "transformed", "001", i,j)