from dtw_score import compute_dtw

reference = "audio_prompts_wav/utt01.wav"
patient = "audio_16k/utt01.wav"

score = compute_dtw(reference, patient)

print("DTW Score:", score)