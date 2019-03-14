from audio_utilities import transcribe_audio_file, record_audio, play_audio
import socket, os
import numpy as np, scipy as sp, matplotlib.pyplot as plt, matplotlib, sklearn, librosa, copy, scipy.io.wavfile,

IP = 'localhost'
PORT = 5050
BUFFER_SIZE = 1024


def main(file_name='file.wav'):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((IP, PORT))
        print('Socket has connected with the socket-server')
    except socket.error, msg:
        print('Could not connect with the socket-server: ' + str(msg))
        sys.exit(1)
        

    # TODO: Get the file name from the file that has been read from javascript
    
    silence_threshold = 0
    
    try:
        silence_threshold = sock.recv(BUFFER_SIZE)
    except Exception as e:
        print('Error while trying to receive silence threshold: ', str(e))
        sys.exit(1)
    
    RESULTS = transcribe_audio_file(file_name)
    transcript, timestamps = get_transcript_and_timestamps(RESULTS)
    audio, sr = librosa.load(file_name)

    # TODO: Replace this with stronger function for augment audio
    _speech, _silence = augment_audio_with_threshold(audio, sr, timestamps, silence_threshold)

    final_audio = []
    speech_counter = 0
    silence_counter = 0
    satisfied = False
    
    if len(silence_chunks) == len(speech_chunks): # silence starts the audio clip
        # print('This audio file begins with a silent period, would you like to remove it (1)\
        # , add it in full (2), or edit it (3).')
        # for now assume the silence is added:
        final_audio.extend(copy.deepcopy(silence_chunks[0]))
        silence_counter += 1
    
    final_audio.extend(copy.deepcopy(speech_chunks[speech_counter]))
    
    while speech_counter < (len(speech_chunks) - 1) or silence_counter < (len(silence_chunks) - 1):
        cur_silence_chunk = copy.deepcopy(silence_chunks[silence_counter])
        
        # satisfied = False
        while True:
            print('Playing the last 5 seconds of the current final audio...')
            
            temp_audio = []
            
            # play 5 seconds of audio or all of it if currently not longer than 5
            if len(final_audio) / sr > 5:
                temp_audio = copy.deepcopy(final_audio[len(final_audio) - (5 * sr):])
            else:
                temp_audio = copy.deepcopy(final_audio)
            
            wavwrite('temp_audio.wav', temp_audio, sr)
            play_audio('temp_audio.wav')
            os.remove('temp_audio.wav')
            
            print('Playing the silence chunk we are currently editing...')
            # play the silence_chunk
            wavwrite('temp_silence.wav', _silence[silence_counter], sr)
            play_audio('temp_silence.wav')
            os.remove('temp_silence.wav')

            
            # shorten the silence_chunk
            sock.send(1)

            # listen for response:
            shorten_ratio = 0
            try:
                shorten_ratio = sock.recv(BUFFER_SIZE)
            except Exception as e:
                print('Error while trying to receive shorten ratio: ', str(e))
                exit()
            
            cur_silence_chunk = copy.deepcopy(cur_silence_chunk[: int(len(cur_silence_chunk) * shorten_ratio)])
    
            # concate to the last x seconds of current recording, append silence, and append
            # next x seconds of next speech chunk
            temp_audio.extend(copy.deepcopy(cur_silence_chunk))
            
            if len(speech_chunks[speech_counter + 1]) > (5 * sr):
                # get only first 5
                temp_audio.extend(copy.deepcopy(speech_chunks[speech_counter + 1][:int(5 * sr)]))
            else:
                temp_audio.extend(copy.deepcopy(speech_chunks[speech_counter + 1]))
            
            # play the new piece
            print('\n\nPlaying how the audio file now sounds at this point...')
            wavwrite('temp_cur_audio.wav', temp_audio, sr)
            play_audio('temp_cur_audio.wav')
            os.remove('temp_cur_audio.wav')

            sock.send(1)

            response = 0
            try:
                response = sock.recv(BUFFER_SIZE)
            except Exception as e:
                print('Error while trying to receive response for satisfaction: ', str(e))
                exit()
            
            # get input on if satisfied
            if response == 1:
                # append final chunks and break
                final_audio.extend(copy.deepcopy(cur_silence_chunk))
                final_audio.extend(copy.deepcopy(speech_chunks[speech_counter + 1]))
                speech_counter += 1
                silence_counter += 1
                break
    
    wavwrite('./edited_audio.wav', final_audio, sr)
    print('Exiting....')
    

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


class NonSpeechType(Enum):
    "Represents if non-speech audio is silence or non-silence"
    SILENCE = auto()
    NOT_SILENCE = auto()

class NonSpeechData:
    sound_type: NonSpeechType
    data: [int]
    def __init__(self, d, st: NonSpeechType):
        self.data = d
        self.sound_type = st


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
                silence_chunks.append(NonSpeechData(audio[: int(threshold * sr)], NonSpeechType.SILENCE))
            else:
                silence_chunks.append(NonSpeechData(audio[: int(speech_timestamps[0].start_time * sr)], NonSpeechType.SILENCE))
        
        # non-speech audio is in the beginning chunk
        else:
            silence_chunks.append(NonSpeechData(audio[:int(speech_chunks[0].start_time * sr)], NonSpeechType.NOT_SILENCE))
        
    
    while end < len(speech_timestamps):
        if speech_timestamps[end].start_time - speech_timestamps[end-1].end_time > threshold:
            h_timestamp = check_if_hesitation_in_between(hesitation_timestamps,
                                                         speech_timestamps[end-1].end_time,
                                                        speech_timestamps[end].start_time)
            speech_chunks.append(audio[int(speech_timestamps[start].start_time * sr): int(speech_timestamps[end-1].end_time * sr)])
            
            if h_timestamp:
                silence_chunks.append(NonSpeechData(audio[int(speech_timestamps[end-1].end_time * sr) : int(speech_timestamps[end].start_time * sr)], NonSpeechType.NOT_SILENCE))
            else:
                silence_chunks.append(NonSpeechData(audio[int(speech_timestamps[end-1].end_time * sr): int(speech_timestamps[end].start_time * sr)][: int(threshold * sr)], NonSpeechType.SILENCE))
            
            start = end
        
        end += 1
    
    # append the final speech chunk
    speech_chunks.append(audio[int(speech_timestamps[start].start_time * sr) :])
    
    # TODO: The last chunks whether silence or non-speech audio gets tossed out
    return speech_chunks, silence_chunks


if __name__ == '__main__':
    main()


#############################################################################
# splits audio into speech and silence chunks with a specified threshold for silence length
# def augment_audio_with_threshold_new(audio, sr, timestamps, threshold):
#     if len(timestamps) < 1:
#         return
#     speech_chunks = []
#     silence_chunks = []    
#     start = 0
#     end = 1
    
#     # NB: If the silence is too short it gets thrown out
#     if timestamps[0].start_time > threshold:
#         silence_chunks.append(SilenceOrFiller(audio[:int(timestamps[0].start_time * sr)], True))
    
#     if timestamps[0].word == '%HESITATION%':
#         silence_chunks.append(SilenceOrFiller(audio[int(timestamps[0].start_time * sr): int(timestamps[1].end_time * sr)], False))
    
#     while end < len(timestamps):
#         if timestamps[end].start_time - timestamps[end - 1].end_time > threshold:
#             speech_chunks.append(audio[int(timestamps[start].start_time * sr): int(timestamps[end-1].end_time * sr)])
#             silence_chunks.append(audio[int(timestamps[end-1].end_time * sr): int(timestamps[end].start_time * sr)])
#             start = end
        
#         end += 1
#     speech_chunks.append(audio[int(timestamps[start].start_time * sr) :])
    
#     return speech_chunks, silence_chunks

# splits audio into speech and silence chunks with a specified threshold for silence length
# def augment_audio_with_threshold(audio, sr, timestamps, threshold):
#     speech_chunks = []
#     silence_chunks = []    
#     start = 0
#     end = 1
    
#     # NB: If the silence is too short it gets thrown out
#     if timestamps[0].start_time > threshold:
#         silence_chunks.append(audio[:int(timestamps[0].start_time * sr)])
    
#     while end < len(timestamps):
#         if timestamps[end].start_time - timestamps[end - 1].end_time > threshold:
#             speech_chunks.append(audio[int(timestamps[start].start_time * sr): int(timestamps[end-1].end_time * sr)])
#             silence_chunks.append(audio[int(timestamps[end-1].end_time * sr): int(timestamps[end].start_time * sr)])
#             start = end
        
#         end += 1
#     speech_chunks.append(audio[int(timestamps[start].start_time * sr) :])
    
#     return speech_chunks, silence_chunks
    


