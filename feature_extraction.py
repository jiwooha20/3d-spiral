import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os


def extract_spectrogram(audio, sr, file_name, save_dir, type="npy"):
    audio = librosa.stft(audio)
    spectrogram = librosa.amplitude_to_db(np.abs(audio), ref=np.max)
    
    if (type=="npy"):
        save_path = save_dir+file_name+"_spec.npy"
        np.save(save_path, spectrogram)
        return
    
    if (type=="jpg"):
        save_path = save_dir+file_name+"_spec.jpg"
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram, sr=sr)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return
        

def extract_chromagram_stft(audio, sr, file_name,save_dir, type="npy"):
    chromagram = librosa.feature.chroma_stft(y=audio, sr=sr)
    
    if (type=="npy"):
        save_path = save_dir+file_name+"_chro.npy"
        np.save(save_path, chromagram)
        return
    
    if (type=="jpg"):
        save_path = save_dir+file_name+"_chro.jpg"
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(chromagram, sr=sr)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return


def extract_mfcc(audio, sr, file_name, save_dir, type="npy"):
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
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return
        
