from pytube import YouTube
from pytube import Playlist
import os
import moviepy.editor as mp
import re
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import glob

# Download the Youtube playlist for original songs and save each song as an mp3 in a folder 
def download_originals(playlist_url, save_path):
    counter = 1
    for url in playlist_url:
        if counter in []:
            print(counter)
            counter += 1
            continue
        else:
            YouTube(url).streams.first().download(save_path, filename_prefix=str(counter) + ' ')
            print(counter)
            counter += 1

    for file in os.listdir(save_path):
        if re.search('mp4', file):
            mp4_path = os.path.join(save_path, file)
            mp3_path = os.path.join(save_path, os.path.splitext(file)[0]+'.mp3')
            new_file = mp.AudioFileClip(mp4_path)
            new_file.write_audiofile(mp3_path)
            os.remove(mp4_path)

# Download the Youtube playlist for mashup songs and save each song as an mp3 in a folder 
def download_originals(playlist_url, save_path):
    playlist = Playlist(playlist_url)
    counter = 1
    seen = 0
    for url in playlist:
        if seen == 0:
            YouTube(url).streams.first().download(save_path, filename_prefix=str(counter) + ' ' + seen + ' ')
            print(counter)
            seen = 1
        else:
            YouTube(url).streams.first().download(save_path, filename_prefix=str(counter) + ' ' + seen + ' ')
            print(counter)
            seen = 0
            counter += 1

        for file in os.listdir(save_path):
            if re.search('mp4', file):
                mp4_path = os.path.join(save_path, file)
                mp3_path = os.path.join(save_path, os.path.splitext(file)[0]+'.mp3')
                new_file = mp.AudioFileClip(mp4_path)
                new_file.write_audiofile(mp3_path)
                os.remove(mp4_path)

# Upload files to google drive
def upload_folder(folder_path):
    # Handles google drive API authentication 
    g_login = GoogleAuth()
    g_login.LocalWebserverAuth()
    drive = GoogleDrive(g_login)

    # Upload each file in the mashups folder
    os.chdir(folder_path)
    for file in glob.glob("*.mp3"):
        with open(file, "r") as f:
            # Filename
            basename = os.path.basename(f.name)
            filename = str(idx) + ' ' + str(basename)
            # Creates file
            new_file = drive.CreateFile({ 'title': str(filename) })
            new_file.SetContentFile(str(basename))
            new_file.Upload() # Files.insert()
            idx +=1
            print("File " + filename + " uploaded")
