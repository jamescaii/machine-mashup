#!/bin/bash
cd ../data/mashup-mp3
for file in *.mp3; do
   ffmpeg -i "$file" -acodec pcm_s16le -ac 1 -ar 44100 ../mashup-wav/"${file%.mp3}".wav
done
cd ..
cd original-mp3
for file in *.mp3; do
   ffmpeg -i "$file" -acodec pcm_s16le -ac 1 -ar 44100 ../original-wav/"${file%.mp3}".wav
done
