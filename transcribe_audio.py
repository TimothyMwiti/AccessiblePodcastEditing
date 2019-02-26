from watson_developer_cloud import SpeechToTextV1
import json

def transcribe_audio_file(path_to_file):
    speech_to_text = SpeechToTextV1(
        iam_apikey='',
        url=''
        )

    with open(path_to_file,
                    'rb') as audio_file:
        speech_recognition_results = speech_to_text.recognize(
            audio=audio_file,
            content_type='audio/mp3',
            timestamps=True
        ).get_result()
    return speech_recognition_results
    
    