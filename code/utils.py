import os
import numpy as np
import math
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
import librosa

SAMPLE_RATE = 8000
WINDOW_LENGTH = 512
NFFT = 512
MAG_MAX = 80

def generate_spectrogram(file_path_in, file_path_out, plot_title):
    """
    Generates the spectrogram corresponding to a signal array

    :param file_path_in: file path of the signal array
    :param file_path_out: file path of the to be generated spectrogram
    :param plot_title: title of the spectrogram
    :return: None
    """

    # Load in signal array
    aud_data = np.load(file_path_in)[:,0]
    # Denormalize signal magnitudes
    aud_data[0] = aud_data[0] * 60
    if (len(aud_data.shape) == 3):
        # Mashup songs
        magnitude = np.transpose(aud_data[0])
        phase = np.transpose(aud_data[1])
        # Calculate original audio signal values
        transformed = magnitude * phase
        signal = librosa.istft(transformed, win_length=WINDOW_LENGTH)
        # Retransform to get sparser data
        transformed = np.abs(librosa.stft(signal))
        # Format spectrogram data into an image using matplotlib
        fig, ax = plt.subplots()
        fig.set_size_inches(30, 5)
        img = librosa.display.specshow(librosa.amplitude_to_db(transformed, ref=np.max), y_axis='log', x_axis='time', ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set(title=plot_title)
        fig.savefig(file_path_out + ".png")
    else:
        # Original songs
        for i in range(2):
            magnitude = np.transpose(aud_data[0,i])
            phase = np.transpose(aud_data[1,i])
            # Calculate original audio signal values
            transformed = magnitude * phase
            signal = librosa.istft(transformed, win_length=WINDOW_LENGTH)
            # Retransform to get sparser data
            transformed = np.abs(librosa.stft(signal))
            # Format spectrogram data into an image using matplotlib
            fig, ax = plt.subplots()
            fig.set_size_inches(30, 5)
            img = librosa.display.specshow(librosa.amplitude_to_db(transformed, ref=np.max), y_axis='log', x_axis='time', ax=ax)
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            ax.set(title=plot_title+" "+str(i+1))
            fig.savefig(file_path_out + "_" + str(i+1) + ".png")

def generate_audio(file_path_in, file_path_out):
    """
    Generates the wav file corresponding to the given signal array

    :param file_path_in: file path of the signal array
    :param file_path_out: file path of the to be generated wav
    :return: None
    """

    # Load in signal array
    aud_data = np.load(file_path_in)[:,0]
    # Denormalize signal magnitudes
    aud_data[0] = aud_data[0] * 60
    if (len(aud_data.shape) == 3):
        # Mashup song
        magnitude = np.transpose(aud_data[0])
        phase = np.transpose(aud_data[1])
        # Calculate original audio signal values
        transformed = magnitude * phase
        signal = librosa.istft(transformed, win_length=WINDOW_LENGTH)
        # Use soundfile to write signal to a wav file
        sf.write(file_path_out + ".wav", signal, SAMPLE_RATE)
    else:
        # Original songs
        for i in range(2):
            magnitude = np.transpose(aud_data[0,i])
            phase = np.transpose(aud_data[1,i])
            # Calculate original audio signal values
            transformed = magnitude * phase
            signal = librosa.istft(transformed, win_length=WINDOW_LENGTH)
            # Use soundfile to write signal to a wav file
            sf.write(file_path_out + "_" + str(i+1) + ".wav", signal, SAMPLE_RATE)

def upload_file(drive, file_path):
    with open(file_path, "r") as f:
        # Get filename
        filename = os.path.basename(f.name)
        # Creates file for Drive
        new_file = drive.CreateFile({ 'title': str(filename) })
        new_file.SetContentFile(str(file_path))
        new_file.Upload()
        print("File " + filename + " uploaded")

def download_folder(drive, folder_id, download_path):
    file_list = drive.ListFile({'q': "'" + folder_id + "' in parents"}).GetList()
    os.chdir(download_path)

    for f in file_list:
        fname = f['title']
        f_ = drive.CreateFile({'id': f['id']})
        f_.GetContentFile(fname)
    os.chdir("../../code")
