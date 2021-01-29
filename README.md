## Machine Mashups: A Transformer-Based Approach to Song Combination
By: James Cai, Kate Nelson, David Inho Lee, Amanda Lee

### Introduction
The motivation for this project stemmed from watching and listening to mashups of popular songs on Youtube. A mashup song is a blend of two or more songs, spliced and superimposed to create a new piece of music. Whether it is a Ed Sheeren x Taylor Swift mix or BTS x Lauv song, these unique combinations can be fun to listen to and work surprisingly well together. Furthermore, producing mashups is a popular hobby or career — professional musicians and amateur content creators alike invest up to days, or even weeks, to create just one mashup. 
 
Deep learning has been applied to mashing images together, as in GANs-based face-morphing, and to generating original music by applying LSTMS to existing songs. This project combines these ideas to explore mashup music creation using a parallelized, transformer-based approach. Spectrogram representations of several thousand Youtube songs are used to ultimately produce “mashed up” audio.

### Data
The data consists of mp3 audio files that originated on Youtube. Mashup songs, originally created manually by Youtubers, served as the labels. For each label, there are two  corresponding inputs — the individual songs from which the mashup was created. For instance, the two inputs for a Bruno Mars x Ed Sheeran mashup song label are the individual Bruno Mars song and Ed Sheeran song.
 
To obtain each piece of data from Youtube, scripts using various Python packages and Google Drive APIs were used. The Pytube and MoviePy packages were employed to download entire Youtube playlists of songs and discard the video data to retain only the audio files. Google Drive APIs were then leveraged to store the data.
 
To efficiently download sufficient amounts of data, Youtube channels consisting entirely of mashup songs were targeted. For instance, the Youtube content creator LopanMashups has over 600 mashups on his channel. Half a dozen similar Youtube channels were used to obtain the label data. For input data, the team manually created Youtube playlists containing the individual songs that the mashup songs were composed from. 
 
The same Python scripts to download the data, segregate the audio files, and upload it to Google Drive were run on the label and input data. During this process, the label data files (mashup songs) were named using the format “[song_index][song_name].mp3”. The input songs (original songs) were named in the form “[song_index][original_index][song_name].mp3”. The [original_index] value is to differentiate between the two input songs that correspond to the same label.
 
Ultimately, the model utilized a total of 2000 mashup song labels, with 4000 inputs.

### Preprocessing
First, the input and label data was converted from mp3 into wav files. This was to enable compatibility with the Librosa Python library, used later in preprocessing. Because each audio file varied in length, they were all trimmed to be a uniform 60 seconds long. 
 
The Librosa package’s short-time fourier transform function was then used to extract the magnitude signal and the phase signal from each wav file as 2D numpy arrays. 
 
The magnitude and phase 2D arrays for all the mashup songs were then combined into one 4D array with the structure [(phase or magnitude), song_index]. The magnitude data corresponds to an index of 0 and phase to an index of 1. The final data structure of the input songs is similar, but with indices [(phase or magnitude), song_index, original_index]. 
 
Additionally, the team also created a function that takes in the magnitude and phase arrays to reconstruct the spectrogram for a wav file. This was later used in visualization of the model’s results. 
 
### Model
The model consists of two parallel transformer-based LSTM architectures, side-by-side. One architecture takes in the two input song magnitude signals, and the other takes in the two input song phase signals. Each of these are shaped (n_samples, width, n_timesteps). Both the magnitude and phase data are represented as vectors at each timestep.
The magnitude array is passed through an encoder LSTM layer of embedding size 512. The output at each timestep is then transferred through an Attention layer, precisely as it was in the course assignment, again with an embedding size of 512. This is done for both input songs, and the outputs of the Attention layer are concatenated together and with the original arrays, then passed into a decoder LSTM layer. The final state of the encoder layers are passed in as the initial state. A final dense layer transforms the output at each timestep to be of sufficient size. This same process is applied, in parallel, to the phase data, and the final “mashed up” magnitudes and phases are combined and converted into mp3 audio.
 
### Results
The model outputs a spectrogram, which represents the mashed up song. When comparing the output to the two input spectrograms, there are certain similarities such as the quieter audio near the beginning and visible beats throughout both spectrograms. The model successfully predicts spectrograms that have some musical or beatlike components, illustrated by the periodic peaks in the magnitude of frequencies that occur at a single time step. 

When the spectrograms files back to mp3 format, the mashed up songs did not resemble music to the naked ear. They do, however, output distinct noises. 
 
### Challenges
Our first challenge was data collection and labeling. While there is a sufficient number of songs on Youtube for training, we underestimated the time required to compile the corresponding pairs of input songs and label them. As a result, the model trained on less data than we had hoped, and it had difficulty extrapolating what it had learned to new data that it had not trained on. 

Another difficulty was our model architecture itself. Originally, we planned on running spectrogram matrices through an image recognition model. However, we realized that this architecture and our project goal were misaligned — training on the “pixels” of each spectrogram did not make sense for audio data. We pivoted to build our model using Transformers and LSTMs. While this is a more logical structure, it’s still an imperfect match, as the intended usage of Transformers and LSTMs is not for audio data. 

The final major challenge we encountered was the difficulty of decreasing the loss for the phase signals. After many modifications to our hyperparameters, we were able to get our loss down to around 0.5. We’re confident that this could be vastly improved with the addition of more training data. 

### Reflection
Overall, we were moderately satisfied with what our model produced. We were indeed able to train the magnitude signals of our audio file, but had trouble with training the phase signals. We also met our base goal of generating some sort of sound from our model given two input songs. We somewhat met the target goal, since our outputted spectrogram and audio file showed resemblance to the beat and magnitude of both input songs. 

Our model worked largely in the way that we had planned. Its primary weakness is its loss value for the phase data of the two input songs. While the loss value of 0.51 is still significant, it is still a vast improvement upon the initial loss values, which were highly unstable or stagnant after one batch, before we tweaked the hyperparameters.

To dive deeper into our struggle with a suitable model architecture, our first approach was to use a model that would train on the images of the spectrograms of the two input songs against the labeled mashup song. After we decided to pivot for the reasons articulated in the previous section, we discovered that a spectrogram of an audio file was essentially composed of the magnitude and the phase signal. We then decided to split our architecture into two: one would handle the magnitude signal, the other the phase signal, and the outputs of both would form our spectrogram and audio file at the end. One future direction of this project is deeper investigation into deep learning models more commonly used for processing sound data. 

Another future direction is definitely gathering more data. As stated above, if we had more time, more data would likely greatly improve the model’s ability to extrapolate patterns in mashup songs. Furthermore, with more time flexibility, we would increase the number of dense layers and continue to experiment with the hyperparameters to improve the loss.

Another interesting direction, suggested by our mentor, is to produce a mask-like output in addition to the mp3 result. This mask matrix would illustrate how much of each input song is heard at each timestamp. For instance, a number closer to 0 could indicate that input song 1 is heavily featured at that timestamp. Correspondingly, a number closer to 1 would designate input song 2 as the primary sound.

The biggest takeaway from this project is the difficulty of creating a model to solve a problem we had not encountered in class. This was a major reason for choosing a project of this scale — we hoped to adapt our learnings to tackle a unique architecture — but it required more brainstorming to find the best model for our solution. Another takeaway is the difficulty of collecting data. The course assignments all had well-curated datasets. Now that we’ve experienced the pain of compiling one, we won’t be taking clean, public datasets for granted in the future!

Overall, our team had a great time working on this project — it allowed us to collaborate in a way we hadn’t gotten a chance to previously in this course, and we learned a lot from each other about the design process and how sound data works. Also a massive shoutout to our TA mentor, Gene, for his willingness to offer suggestions and answer questions!

### Works Cited
Frenzel, M. (2019, February 08). NeuralFunk - Combining Deep Learning with Sound Design. Retrieved November 11, 2020, from https://towardsdatascience.com/neuralfunk-combining-deep-learning-with-sound-design-91935759d628
Hebbar, R. (2017, November 28). Music2vec: Generating Vector Embeddings for Genre-Classification Task. Retrieved November 11, 2020, from https://medium.com/@rajatheb/music2vec-generating-vector-embedding-for-genre-classification-task-411187a20820
WaveNet: A Generative Model for Raw Audio. (n.d.). Retrieved November 11, 2020, from https://deepmind.com/blog/article/wavenet-generative-model-raw-audio
