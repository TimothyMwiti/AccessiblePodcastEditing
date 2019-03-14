from watson_developer_cloud import SpeechToTextV1
import json
import pyaudio
import wave


FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024

def transcribe_audio_file(path_to_file='./file.wav'):
    speech_to_text = SpeechToTextV1(
        iam_apikey='',
        url='https://stream.watsonplatform.net/speech-to-text/api'
        )

    with open(path_to_file,
                    'rb') as audio_file:
        speech_recognition_results = speech_to_text.recognize(
            audio=audio_file,
            content_type='audio/wav',
            timestamps=True
        ).get_result()
    return speech_recognition_results


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


def record_audio(filename='file.wav', duration=30):
    RECORD_SECONDS = duration
    WAVE_OUTPUT_FILENAME = filename
    audio = pyaudio.PyAudio()
    
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    print("recording...")
    frames = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")
    
    
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


# print(transcribe_audio_file('./reading_article_with_silences.wav'))

if __name__ == '__main__':
    # play_audio('edited_audio.wav')
    record_audio('hesitations_test2.wav', 45)
    # play_audio('hesitations_test.wav')