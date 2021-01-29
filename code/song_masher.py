import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from utils import *
from model import SongMasher
import sys
import random
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def train(model, train_originals, train_mashes):
    """
	Runs through one epoch - all training examples.

	:param model: the initialized model to use for forward and backward pass
	:param train_originals: training data, list of the original two songs that were used to create the corresponding mash 
                            of shape (num_examples, 2, n_timesteps)
	:param train_mashes: training labels, list of the mashed songs derived from the corresponding two training data songs
                         of shape (num_examples, n_timesteps)
	:return: None
	"""

    # Shuffle inputs
    idx = np.arange(train_originals.shape[0])
    idx = tf.random.shuffle(idx)
    train_originals = tf.gather(train_originals, idx)
    train_originals1 = train_originals[:,0,:,:]
    train_originals2 = train_originals[:,1,:,:]
    train_mashes = tf.gather(train_mashes, idx)
    # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=model.learning_rate)

    # Iterate through each batch
    for i in range(0, train_originals.shape[0], model.batch_size):
        # Get batch of data
        orig_batch1 = tf.cast(train_originals1[i:i+model.batch_size], np.float32)
        orig_batch2 = tf.cast(train_originals2[i:i+model.batch_size], np.float32)
        mash_batch = tf.cast(train_mashes[i:i+model.batch_size], np.float32)

        # Calculate predictions and loss
        with tf.GradientTape() as tape:
            artif_mashes = model([orig_batch1, orig_batch2])
            artif_mashes = tf.reshape(artif_mashes, [artif_mashes.shape[0], artif_mashes.shape[1] * artif_mashes.shape[2]])
            mash_batch = tf.reshape(mash_batch, [mash_batch.shape[0], mash_batch.shape[1] * mash_batch.shape[2]])
            loss = model.loss_function(artif_mashes, mash_batch)
            print("Batch " + str(int(i/model.batch_size)) + " Loss: %.6f" % (loss.numpy()), flush=True)

        # Apply gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_originals, test_mashes):
    """
    Runs through one epoch - all testing examples.

    :param model: the initialized model to use for forward and backward pass
    :param test_originals: testing data, list of the original two songs that were used to create the corresponding mash 
                           of shape (num_examples, 2, n_timesteps)
    :param test_mashes: testing labels, list of the mashed songs derived from the corresponding two training data songs
                        of shape (num_examples, n_timesteps)
    :return: Average batch loss of the model on the testing set
    """

    # Shuffle inputs
    idx = np.arange(test_originals.shape[0])
    idx = tf.random.shuffle(idx)
    test_originals = tf.gather(test_originals, idx)
    test_originals1 = test_originals[:,0,:,:]
    test_originals2 = test_originals[:,1,:,:]
    test_mashes = tf.gather(test_mashes, idx)

    losses = []
    # Iterate through each batch
    for i in range(0, test_originals.shape[0], model.batch_size):
        # Get batch of data
        orig_batch1 = tf.cast(test_originals1[i:i+model.batch_size], np.float32)
        orig_batch2 = tf.cast(test_originals2[i:i+model.batch_size], np.float32)
        mash_batch = tf.cast(test_mashes[i:i+model.batch_size], np.float32)

        # Calculate predictions and loss
        artif_mashes = model([orig_batch1, orig_batch2])
        artif_mashes = tf.reshape(artif_mashes, [artif_mashes.shape[0], artif_mashes.shape[1] * artif_mashes.shape[2]])
        mash_batch = tf.reshape(mash_batch, [mash_batch.shape[0], mash_batch.shape[1] * mash_batch.shape[2]])
        losses.append(model.loss_function(artif_mashes, mash_batch))
    return np.average(losses)

def visualize_testing_example(magnitude_model, phase_model, test_orig_mag, test_orig_pha, test_mash_mag, test_mash_pha, index):
    # Create numpy arrays to be fed into visualization functions
    orig = np.array([[test_orig_mag[index]], [test_orig_pha[index]]])
    np.save("../data/test/orig_testn_" + str(index), orig)
    mash = np.array([[test_mash_mag[index]], [test_mash_pha[index]]])
    np.save("../data/test/mash_testn_" + str(index), mash)

    # Create numpy arrays for model generated magnitude and phase
    artif_mag = magnitude_model([orig[0,:,0], orig[0,:,1]])
    artif_pha = phase_model([orig[1,:,0], orig[1,:,1]])
    artif = np.array([artif_mag, artif_pha])
    np.save("../data/test/artif_testn_" + str(index), artif)

    # Create spectrograms for original songs, mashed song, and model-produced song
    generate_spectrogram("../data/test/orig_testn_" + str(index) + ".npy", "../data/test/orig_spect_testn_" + str(index), "Originals Spectrogram")
    generate_spectrogram("../data/test/mash_testn_" + str(index) + ".npy", "../data/test/mash_spect_testn_" + str(index), "Mashup Spectrogram")
    generate_spectrogram("../data/test/artif_testn_" + str(index) + ".npy", "../data/test/artif_spect_testn_" + str(index), "Model-Produced Spectrogram")

    # Create audio files for original songs, mashed song, and model-produced song
    generate_audio("../data/test/orig_testn_" + str(index) + ".npy", "../data/test/orig_song_testn_" + str(index))
    generate_audio("../data/test/mash_testn_" + str(index) + ".npy", "../data/test/mash_song_testn_" + str(index))
    generate_audio("../data/test/artif_testn_" + str(index) + ".npy", "../data/test/artif_song_testn_" + str(index))

def visualize_unseen_example(magnitude_model, phase_model, wav_path):
    # Convert desired input songs to magnitude and phase arrays
    convert_original_to_array(wav_path, wav_path)
    mag_in, pha_in = get_data(wav_path, wav_path, 1)
    # Pass them through the model to generate the mashup
    mag_out = magnitude_model([mag_in[:,0], mag_in[:,1]])
    pha_out = phase_model([pha_in[:,0], pha_in[:,1]])
    artif = np.array([mag_out, pha_out])
    np.save("../data/test/artif", artif)

    # Create spectrograms for original songs and model-produced song
    generate_spectrogram(wav_path + "original.npy", wav_path + "original", "Original Spectrogram")
    generate_spectrogram(wav_path + "artif.npy", wav_path + "artif", "Model-Produced Spectrogram")

    # Create audio files for original songs and model-produced song
    generate_audio(wav_path + "original.npy", wav_path + "original")
    generate_audio(wav_path + "artif.npy", wav_path + "artif")

def main():
    # Set up Google Drive authentication
    g_auth = GoogleAuth()
    g_auth.LocalWebserverAuth()
    drive = GoogleDrive(g_auth)
    # Download mp3s
    print("Downloading...", flush=True)
    download_folder(drive, "1EbwrLZxZGOvLTuGPPWmrKlYjbS_UuNw_", "../data/original-mp3")
    download_folder(drive, "1dPEIZhRvM-YeKZPgOZKRNhJQR5UyVptH", "../data/mashup-mp3")

    # Preprocess data
    print("Running preprocessing...", flush=True)
    prep()
    print("Uploading...", flush=True)
    upload_file(drive, "../data/preprocessed/original.npy")
    upload_file(drive, "../data/preprocessed/mashup.npy")

    # Gather preprocessed training and testing data
    print("Gathering data...", flush=True)
    train_orig_mag, train_orig_pha, train_mash_mag, train_mash_pha, test_orig_mag, test_orig_pha, \
        test_mash_mag, test_mash_pha = get_data("../data/preprocessed/original.npy", "../data/preprocessed/mashup.npy", 0.95)

    test_orig_mag = train_orig_mag
    test_orig_pha = train_orig_pha
    test_mash_mag = train_mash_mag
    test_mash_pha = train_mash_pha
    
    # Create models for both the magnitude and phase of the signal
    print("Training...", flush=True)
    magnitude_model = SongMasher(train_orig_mag.shape[2], train_orig_mag.shape[3], 0.001)
    phase_model = SongMasher(train_orig_pha.shape[2], train_orig_pha.shape[3], 0.05)
    # Train and test model for 100 epochs.
    for epoch in range(100):
        train(magnitude_model, train_orig_mag, train_mash_mag)
        train(phase_model, train_orig_pha, train_mash_pha)
        mag_loss = test(magnitude_model, test_orig_mag, test_mash_mag)
        pha_loss = test(phase_model, test_orig_pha, test_mash_pha)
        mag_loss = test(magnitude_model, train_orig_mag, train_mash_mag)
        pha_loss = test(phase_model, train_orig_pha, train_mash_pha)
        print("Epoch %d Mag Test Loss: %.6f" % (epoch, mag_loss), flush=True)
        print("Epoch %d Pha Test Loss: %.6f" % (epoch, pha_loss), flush=True)
    
    # Save models after done training
    print("Saving models...", flush=True)
    magnitude_model.save('../model/magnitude_model')
    phase_model.save('../model/phase_model')
    # Load models for visualization
    print("Loading models...", flush=True)
    mag_model = tf.keras.models.load_model("../model/magnitude_model")
    pha_model = tf.keras.models.load_model("../model/phase_model")
    # Visualize one example from the testing set
    print("Visualizing models...", flush=True)
    visualize_testing_example(mag_model, pha_model, test_orig_mag, test_orig_pha, test_mash_mag, test_mash_pha, 0)
    # Visualize one example from the training set
    visualize_testing_example(mag_model, pha_model, train_orig_mag, train_orig_pha, train_mash_mag, train_mash_pha, 1)
    # Upload visualized examples
    print("Uploading results...", flush=True)
    test_files = ["artif_song_testn_0.wav", "artif_spect_testn_0.png", "artif_testn_0.npy", "mash_song_testn_0.wav", "mash_spect_testn_0.png", "mash_testn_0.npy", 
        "orig_song_testn_0_1.wav", "orig_song_testn_0_2.wav", "orig_spect_testn_0_1.png", "orig_spect_testn_0_2.png", "orig_testn_0.npy"]
    for fname in test_files:
        upload_file(drive, "../data/test/" + fname)


if __name__ == '__main__':
    main()
