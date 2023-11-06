import os
import math
import pretty_midi

from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
import audioop

def loudness_to_normalized(loudness):
    return 10 ** (loudness / 20)

def main():
    directory_vevo_chord = "../dataset/vevo_chord/lab_v2_norm/all/"
    directory_vevo_wav = "../dataset/vevo_audio/wav/"

    for filename in sorted(os.listdir(directory_vevo_chord)):
        chord_filepath = os.path.join( directory_vevo_chord, filename)
        print(chord_filepath)
        ct = 0
        with open(chord_filepath, encoding = 'utf-8') as f:
            for line in f:
                line = line.strip()
                line_arr = line.split(" ")
                if len(line_arr) > 1:
                    ct = ct+1

        fname = filename.split(".")[0]
        wavpath = os.path.join(directory_vevo_wav, filename.replace("lab", "wav"))

        audio_data = AudioSegment.from_file(wavpath)
        audio_data = audio_data.set_channels(1)  # convert to mono
        audio_data = audio_data.set_frame_rate(44100)  # set sample rate to 44100 Hz
        chunk_length = 1000  # chunk length in milliseconds
        chunks = make_chunks(audio_data, chunk_length)
        loudness_per_second = []
        for chunk in chunks:
            data = chunk.raw_data  # get raw data as bytes
            rms = audioop.rms(data, 2)  # calculate RMS loudness using audioop module
            loudness = 20 * np.log10(rms / 32767)  # convert to decibels
            normalized_loudness = loudness_to_normalized(loudness)  # convert to 0-1 scale
            normalized_loudness = format(normalized_loudness, ".4f")
            loudness_per_second.append(normalized_loudness)
        
        fpathname = "../dataset/vevo_loudness/all/" + fname + ".lab"
        with open(fpathname, 'w', encoding = 'utf-8') as f:
            for i in range(0, len(loudness_per_second)):
                f.write(str(i) + " "+str(loudness_per_second[i])+"\n")

if __name__ == "__main__":
    main()

