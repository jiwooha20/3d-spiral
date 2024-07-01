import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os


def extract_spectrogram(y, sr, file_path,save_dir, type="npy"):

    #y, sr = librosa.load(file_path, sr=None)
    #print(file_path, "sampling rate is", sr)
    y = librosa.stft(y)
    spectrogram = librosa.amplitude_to_db(np.abs(y), ref=np.max)
    
    if (type=="npy"):
        save_path = save_dir+os.path.splitext(os.path.basename(file_path))[0]+"_spec.npy"
        np.save(save_path, spectrogram)
        return
    
    if (type=="jpg"):
        save_path = save_dir+os.path.splitext(os.path.basename(file_path))[0]+"_spec.jpg"
        plt.figure(figsize=(10, 4))
#             librosa.display.specshow(spectrogram, sr=sr)
        librosa.display.specshow(spectrogram, y_axis='log', x_axis='time', sr=sr)
        plt.axis('off')
        plt.tight_layout()
        # sppectrogram .jpg 저장
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        

def extract_chromagram_stft(y, sr, file_path,save_dir, type="npy"):

    #y, sr = librosa.load(file_path, sr=None)
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    
    if (type=="npy"):
        save_path = save_dir+os.path.splitext(os.path.basename(file_path))[0]+"_chro.npy"
        np.save(save_path, chromagram)
        return
    
    if (type=="jpg"):
        save_path = save_dir+os.path.splitext(os.path.basename(file_path))[0]+"_chro.jpg"
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(chromagram, sr=sr)
        plt.axis('off')
        plt.tight_layout()
        # mfcc.jpg 저장
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return


def extract_mfcc(audio, sr, file_name, save_dir, type="npy"):

    #audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    if (type=="npy"):
        save_path = save_dir+file_name+"_mfcc.npy"
        np.save(save_path, mfcc)
        return
        
    if (type=="jpg"):
        save_path = save_dir+file_name+"_mfcc.jpg"
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, sr=sr)
        plt.axis('off')
        plt.tight_layout()
        # mfcc.jpg 저장
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return
        

if __name__ == "__main__":
    # 다시 짜야됨
    save_dir = "./save_dir/"

    root_dir = "./sound_sample"
    for (root,dirs,files) in os.walk(root_dir):
        if len(files) > 0:
            for file_name in files:
                extract_spectrogram(root_dir+"/"+file_name, save_dir,"jpg")
                extract_chromagram_stft(root_dir+"/"+file_name, save_dir,"npy")
                extract_chromagram_stft(root_dir+"/"+file_name, save_dir,"jpg")
                extract_mfcc(file_name, save_dir,"npy")
                extract_mfcc(file_name, save_dir,"jpg")