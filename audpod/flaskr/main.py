import functools
from . import extras
import numpy as np, scipy as sp, librosa, copy, scipy.io.wavfile, os
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

SILENCE_THRESHOLD = 0.5
bp = Blueprint('main', __name__, url_prefix='/main')


RESULTS = {'results': [{'alternatives': [{'timestamps': [['hi', 0.25, 0.65]], 'confidence': 0.83, 'transcript': 'hi '}], 'final': True}, {'alternatives': [{'timestamps': [['my', 2.21, 2.47], ['name', 2.47, 2.84], ['is', 2.84, 3.22]], 'confidence': 0.93, 'transcript': 'my name is '}], 'final': True}, {'alternatives': [{'timestamps': [['%HESITATION', 4.38, 5.04]], 'confidence': 0.99, 'transcript': '%HESITATION '}], 'final': True}, {'alternatives': [{'timestamps': [['Amir', 6.82, 7.3], ['saja', 7.3, 7.9]], 'confidence': 0.19, 'transcript': 'Amir saja '}], 'final': True}, {'alternatives': [{'timestamps': [['I', 9.66, 10.11], ['am', 10.17, 10.69]], 'confidence': 0.31, 'transcript': 'I am '}], 'final': True}, {'alternatives': [{'timestamps': [['our', 11.91, 12.46]], 'confidence': 0.45, 'transcript': 'our '}], 'final': True}, {'alternatives': [{'timestamps': [['our', 14.58, 14.75], ['first', 14.75, 15.37], ['year', 15.41, 15.72], ['PhD', 15.72, 16.27], ['student', 16.27, 17], ['at', 17.4, 17.87]], 'confidence': 0.9, 'transcript': 'our first year PhD student at '}], 'final': True}, {'alternatives': [{'timestamps': [['%HESITATION', 19.92, 20.57]], 'confidence': 0.98, 'transcript': '%HESITATION '}], 'final': True}, {'alternatives': [{'timestamps': [['Northwestern', 22.39, 23.25], ['University', 23.25, 24.17]], 'confidence': 0.96, 'transcript': 'Northwestern University '}], 'final': True}, {'alternatives': [{'timestamps': [['my', 30.66, 30.89], ['research', 30.89, 31.8]], 'confidence': 0.99, 'transcript': 'my research '}], 'final': True}, {'alternatives': [{'timestamps': [['is', 32.36, 32.55], ['related', 32.55, 33.31], ['to', 33.38, 33.71], ['accessibility', 33.8, 34.84]], 'confidence': 0.97, 'transcript': 'is related to accessibility '}], 'final': True}, {'alternatives': [{'timestamps': [['%HESITATION', 36.59, 37.43]], 'confidence': 0.27, 'transcript': '%HESITATION '}], 'final': True}, {'alternatives': [{'timestamps': [['I', 40.59, 41.04], ['am', 41.08, 41.55], ['taking', 41.71, 42.47]], 'confidence': 0.85, 'transcript': 'I am taking '}], 'final': True}, {'alternatives': [{'timestamps': [['the', 43.35, 43.5], ['mission', 43.5, 44.07], ['perception', 44.07, 44.96]], 'confidence': 0.74, 'transcript': 'the mission perception '}], 'final': True}], 'result_index': 0}
# RESULTS = extras.transcribe_audio_file('./flaskr/audio.wav')

transcript, timestamps = extras.get_transcript_and_timestamps(RESULTS)
audio, sr = librosa.load('./flaskr/audio.wav')

_speech_chunks = []
_silence_chunks = []

FINAL_AUDIO = []
_speech_counter = 0
_silence_counter = 0
_reduce_ratio = 1.0
_last_pressed = None


@bp.route('/index', methods=('GET', 'POST'))
def run_method():
    global _speech_chunks, _silence_chunks, _speech_counter, _silence_counter, FINAL_AUDIO, sr, _last_pressed, _reduce_ratio, timestamps
    if request.method == 'POST':
        error = None
        if request.form['audpod_button'] == "Set Threshold":
            _last_pressed = 'set threshold'
            _silence_threshold = 0.0            
            try:
                _silence_threshold = float(request.form['silenceThreshold'])
            except Exception as e:
                print(e)
                error = 'The silence threshold entered is not in the correct format'
            
            if error is None:
                SILENCE_THRESHOLD = _silence_threshold
                _speech_chunks, _silence_chunks = extras.augment_audio_with_threshold(audio, sr, timestamps, SILENCE_THRESHOLD)

                # TODO: If there's no silence just return since the requirement is met

                if _silence_chunks[_silence_counter].pos != "start": # if silence does not start
                    FINAL_AUDIO.extend(copy.deepcopy(_speech_chunks[_speech_counter]))
                    _speech_counter += 1
                return render_template('main/index.html', silence_counters = [[_silence_counter, len(_silence_chunks)]])
        elif request.form["audpod_button"] == "Play Non-Speech Chunk (L)":
            _last_pressed = 'play silent chunk'
            # print('Playing the current silence chunk that is to be edited')
            if _silence_counter < len(_silence_chunks):
                extras.wavwrite('temp_silence.wav', _silence_chunks[_silence_counter].data, sr)
                extras.play_audio('temp_silence.wav')
                os.remove('temp_silence.wav')
                return render_template('main/index.html', silence_counters = [[_silence_counter, len(_silence_chunks)]])
            else:
                error = 'There are no more silent chunks to play, click finish and download to download the edited file'
        elif request.form["audpod_button"] == "Reduce Non-Speech Chunk (K)":
            _last_pressed = 'reduce silent chunk'
            try:
                _reduce_ratio = float(request.form['reduceRatio'])
            except Exception as e:
                print(e)
                error = 'The reduce ratio entered is not correct. Please re-enter and click the reduce silent chunk button'
            
            if error is None and _silence_counter < len(_silence_chunks):
                temp_audio = []
                if len(FINAL_AUDIO) / sr > 2:
                    temp_audio = copy.deepcopy(FINAL_AUDIO[len(FINAL_AUDIO) - (2 * sr):])
                elif len(FINAL_AUDIO) != 0:
                    temp_audio = copy.deepcopy(FINAL_AUDIO)
                
                cur_silence_chunk = copy.deepcopy(_silence_chunks[_silence_counter].data[: int(len(_silence_chunks[_silence_counter].data) * _reduce_ratio)])
                temp_audio.extend(copy.deepcopy(cur_silence_chunk))

                if _silence_chunks[_silence_counter].pos == "mid" or _silence_chunks[_silence_counter].pos == "start":
                    if len(_speech_chunks[_speech_counter]) > (2 * sr):
                        # get only first 2
                        temp_audio.extend(copy.deepcopy(_speech_chunks[_speech_counter][:int(2 * sr)]))
                    else:
                        temp_audio.extend(copy.deepcopy(_speech_chunks[_speech_counter]))
                
                extras.wavwrite('temp_cur_audio.wav', temp_audio, sr)
                extras.play_audio('temp_cur_audio.wav')
                os.remove('temp_cur_audio.wav')
                return render_template('main/index.html', silence_counters = [[_silence_counter, len(_silence_chunks)]])
            else:
                error = 'No Silent Chunk to Reduce'
        elif request.form["audpod_button"] == "Finish & Download (M)":
            extras.wavwrite('edited_audio.wav', FINAL_AUDIO, sr)
            flash('The file has been downloaded')
            return render_template('main/index.html', silence_counters = [[_silence_counter, len(_silence_chunks)]])
        elif request.form["audpod_button"] == "Confirm Reduced Non-Speech Chunk (H)":
            if _last_pressed != 'reduce silent chunk':
                error = "Please reduce the current silent chunk before proceeding"
            
            if error is None and _silence_counter < len(_silence_chunks):
                cur_silence_chunk = copy.deepcopy(_silence_chunks[_silence_counter].data[: int(len(_silence_chunks[_silence_counter].data) * _reduce_ratio)])
                cur_silence_chunk = cur_silence_chunk * 0.01 #############
                FINAL_AUDIO.extend(copy.deepcopy(cur_silence_chunk))
                if _silence_chunks[_silence_counter].pos == "mid" or _silence_chunks[_silence_counter].pos == "start":
                    FINAL_AUDIO.extend(copy.deepcopy(_speech_chunks[_speech_counter]))
                _silence_counter += 1
                _speech_counter += 1
                if _silence_counter == len(_silence_chunks) :
                	extras.play_audio('./flaskr/beepComplete.wav')
                	extras.play_audio('./flaskr/beepComplete.wav')
                else :
                	extras.play_audio('./flaskr/beep1.wav')

                return render_template('main/index.html', silence_counters = [[_silence_counter, len(_silence_chunks)]])

        flash(error)
        
    return render_template('main/index.html', silence_counters = [[_silence_counter, len(_silence_chunks)]])
