import numpy as np
from pydub import AudioSegment
audio = AudioSegment.from_file('audio.m4a')

print(audio.shape)

