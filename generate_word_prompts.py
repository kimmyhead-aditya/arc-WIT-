from gtts import gTTS
import subprocess
import os

words = [
    'आम', 'पीठ', 'घर', 'पत्ता', 'नाक', 'सड़क', 'दूध', 'ठंड', 'बाल', 'खून',
    'काम', 'ढोल', 'रात', 'भूख', 'आग', 'चाबी', 'धरना', 'हाथ', 'माँ', 'जंगल',
    'शेर', 'फल', 'साँप', 'हल', 'फूल', 'दिन', 'गाल', 'तेल', 'पानी', 'आम'
]

OUTPUT_DIR = 'audio_prompts_wav'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('Generating word prompts...')

for i, word in enumerate(words):
    utt_id = f'utt{i+1:02d}'
    mp3_file = f'{utt_id}.mp3'
    wav_file = os.path.join(OUTPUT_DIR, f'{utt_id}.wav')

    tts = gTTS(word, lang='hi')
    tts.save(mp3_file)

    subprocess.run([
        'ffmpeg',
        '-loglevel', 'quiet',
        '-y',
        '-i', mp3_file,
        '-ar', '16000',
        '-ac', '1',
        wav_file
    ])

    os.remove(mp3_file)
    print(f'  {utt_id} — {word}')

print('Done. All word prompts regenerated.')
