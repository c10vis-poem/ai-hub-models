import onnxruntime as ort
import numpy as np
import subprocess
import scipy.io.wavfile
import sys
import os

KOKORO_DIR = os.path.expanduser('~/kokoro')
MODEL_PATH = os.path.join(KOKORO_DIR, 'onnx', 'model.onnx')
VOICES_DIR = os.path.join(KOKORO_DIR, 'voices')


def phonemize(text, language='en-us'):
    process = subprocess.run(
        ['espeak', '-q', '--ipa', '-v', language, text],
        capture_output=True,
        text=True
    )
    ipa = process.stdout.strip()

    phoneme_map = {
        ' ': 0, 'X': 1, 'ə': 2, 'e': 3, 'I': 4, 'k': 5, 't': 6, 's': 7, 'n': 8,
        'o': 9, 'a': 10, 'd': 11, 'm': 12, 'l': 13, 'r': 14, 'i': 15, 'p': 16,
        'h': 17, 'b': 18, 'z': 19, 'w': 20, 'v': 21, 'u': 22, 'f': 23, 'ɡ': 24,
        'ŋ': 25, 'ʃ': 26, 'j': 27, 'ɔ': 28, 'ð': 29, 'θ': 30, 'ɛ': 31, 'ʒ': 32,
        'æ': 33, 'ʊ': 34, 'aɪ': 35, 'aʊ': 36, 'dʒ': 37, 'eɪ': 38, 'oʊ': 39,
        'ɔɪ': 40, 'tʃ': 41, 'ʌ': 42
    }

    tokens = []
    for p in ipa:
        if p in phoneme_map:
            tokens.append(phoneme_map[p])

    return np.array(tokens, dtype=np.int64).reshape(1, -1)


def synthesize(text, voice_name='am_adam'):
    phoneme_ids = phonemize(text)

    voice_path = os.path.join(VOICES_DIR, f'{voice_name}.bin')
    if not os.path.exists(voice_path):
        raise ValueError(f"Voice '{voice_name}' not found at {voice_path}")

    voice_embedding = np.fromfile(voice_path, dtype=np.float32).reshape(1, -1)

    session = ort.InferenceSession(MODEL_PATH)
    input_names = [inp.name for inp in session.get_inputs()]

    inputs = {
        'input_ids': phoneme_ids,
        'style': voice_embedding,
    }

    if 'speed' in input_names:
        inputs['speed'] = np.array([1.0], dtype=np.float32)

    result = session.run(None, inputs)
    audio_out = result[0].squeeze()

    scipy.io.wavfile.write('output.wav', 24000, audio_out)
    print("Audio saved to output.wav")

    subprocess.run(['ffplay', '-nodisp', '-autoexit', 'output.wav'])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python kokoro_tts.py "<text>" [voice_name]')
        sys.exit(1)

    text_to_synthesize = sys.argv[1]
    voice = sys.argv[2] if len(sys.argv) > 2 else 'am_adam'

    synthesize(text_to_synthesize, voice)
