from enum import Enum, auto
from typing import List, Optional
from watson_developer_cloud import SpeechToTextV1
import json
import pyaudio, wave, numpy as np, scipy as sp, copy, scipy.io.wavfile

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024

def transcribe_audio_file(path_to_file='./file.wav'):
    speech_to_text = SpeechToTextV1(
        iam_apikey='',
        url=''
        )

    with open(path_to_file,
                    'rb') as audio_file:
        speech_recognition_results = speech_to_text.recognize(
            audio=audio_file,
            content_type='audio/wav',
            timestamps=True
        ).get_result()
    return speech_recognition_results


class NonSpeechType(Enum):
    "Represents if non-speech audio is silence or non-silence"
    SILENCE = auto()
    NOT_SILENCE = auto()

class NonSpeechData:
    sound_type: NonSpeechType
    pos = ""
    data: [int]
    def __init__(self, d, st: NonSpeechType, p="mid"):
        self.data = d
        self.sound_type = st
        self.pos = p
    
class WordTimeStamp:
    word : str
    start_time : float
    end_time : float
    
    def __init__(self, word, start, end):
        self.word = word
        self.start_time = start
        self.end_time = end  
    
    def __str__(self):
        return "[" + self.word + ", " + str(self.start_time) + ", " + str(self.end_time) + "]"
    
    def __repr__(self):
        return self.__str__()
    
def wavwrite(filepath, data, sr, norm=True, dtype='int16',):
    '''
    Write wave file using scipy.io.wavefile.write, converting from a float (-1.0 : 1.0) numpy array to an integer array
    
    Parameters
    ----------
    filepath : str
        The path of the output .wav file
    data : np.array
        The float-type audio array
    sr : int
        The sampling rate
    norm : bool
        If True, normalize the audio to -1.0 to 1.0 before converting integer
    dtype : str
        The output type. Typically leave this at the default of 'int16'.
    '''
    if norm:
        data /= np.max(np.abs(data))
    data = data * np.iinfo(dtype).max
    data = data.astype(dtype)
    scipy.io.wavfile.write(filepath, sr, data)

# get the transcript and timestamps
def get_transcript_and_timestamps(results):
    transcript = ""
    timestamps = []
    for r in results['results']:
        transcript += r['alternatives'][0]['transcript']
        cur_time_stamps = r['alternatives'][0]['timestamps']
        
        for t in cur_time_stamps:
            timestamps.append(WordTimeStamp(t[0], t[1], t[2]))
    
    return transcript, timestamps


# gets the timestamps where a hesitation is indicated on the transcript
def get_hesitation_timestamps_from_timestamps(timestamps):
    hesitations = []
    for t in timestamps:
        if t.word == "%HESITATION":
            hesitations.append(t)
    
    return hesitations


def augment_audio_with_threshold(audio, sr, timestamps, threshold):
    # TODO: This assumes that there is hesitation in the speech
    
    hesitation_timestamps = get_hesitation_timestamps_from_timestamps(timestamps)
    speech_timestamps = [x for x in timestamps if x not in get_hesitation_timestamps_from_timestamps(timestamps)]
    
    speech_chunks = []
    non_speech_chunks = []

    start = 0
    end = 1
    
    """
    Beginning of audio edited in the following ways:
        - If it begins with speech straight away, we skip this case
        - If it begins with just silence, the silence is truncated to the threshold
          , labeled as silence and appended to the non-speech chunks.
        - If it begins with just non-speech with audio, we append the full chunk to 
          non-speech chunks and label it as non-speech sound
        - If it begins with both a hesitation and silence, the two are combined and
          labeled as non-speech sound, then appended to the non-speech chunks
    """
    if speech_timestamps[0].start_time >= 0:
        # just silence with no hesitation
        if len(hesitation_timestamps) < 1 or speech_timestamps[0].start_time < hesitation_timestamps[0].start_time:
            if speech_timestamps[start].start_time > threshold:
                non_speech_chunks.append(NonSpeechData(audio[: int(threshold * sr)], NonSpeechType.SILENCE, "start"))
            else:
                non_speech_chunks.append(NonSpeechData(audio[: int(speech_timestamps[0].start_time * sr)], NonSpeechType.SILENCE, "start"))
        
        # non-speech audio is in the beginning chunk
        else:
            non_speech_chunks.append(NonSpeechData(audio[:int(speech_chunks[0].start_time * sr)], NonSpeechType.NOT_SILENCE, "start"))
        
    
    while end < len(speech_timestamps):
        if speech_timestamps[end].start_time - speech_timestamps[end-1].end_time > threshold:
            h_timestamp = check_if_hesitation_in_between(hesitation_timestamps,
                                                         speech_timestamps[end-1].end_time,
                                                        speech_timestamps[end].start_time)
            speech_chunks.append(audio[int(speech_timestamps[start].start_time * sr): int(speech_timestamps[end-1].end_time * sr)])
            
            if h_timestamp:
                non_speech_chunks.append(NonSpeechData(audio[int(speech_timestamps[end-1].end_time * sr) : int(speech_timestamps[end].start_time * sr)], NonSpeechType.NOT_SILENCE, "mid"))
            else:
                non_speech_chunks.append(NonSpeechData(audio[int(speech_timestamps[end-1].end_time * sr): int(speech_timestamps[end].start_time * sr)][: int(threshold * sr)], NonSpeechType.SILENCE, "mid"))
            
            start = end
        
        end += 1
    
    # append the final speech chunk
    speech_chunks.append(audio[int(speech_timestamps[start].start_time * sr) :])
    
    # TODO: The last chunks whether silence or non-speech audio gets tossed out
    return speech_chunks, non_speech_chunks


def check_if_hesitation_in_between(h_timestamps, start_time, end_time) -> Optional[WordTimeStamp]:
    h_t = None
    
    for h in h_timestamps:        
        if h.start_time >= start_time and h.end_time <= end_time:
            return h
    return None

def play_audio(filename='file.wav'):
    chunk = CHUNK
    wf = wave.open(filename, 'rb')
    # create an audio object
    p = pyaudio.PyAudio()

    # open stream based on the wave object which has been input.
    stream = p.open(format =
                    p.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = wf.getframerate(),
                    output = True)

    # read data (based on the chunk size)
    data = wf.readframes(CHUNK)
    
    # play stream (looping from beginning of file to the end)
    while data != '' and len(data) != 0:
        # writing to the stream is what *actually* plays the sound.
        stream.write(data)
        data = wf.readframes(chunk)
    
    # cleanup stuff.
    stream.close()    
    p.terminate()
    return