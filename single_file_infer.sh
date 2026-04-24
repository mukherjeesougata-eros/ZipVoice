python3 -m zipvoice.bin.infer_zipvoice \
  --model-name zipvoice \
  --model-dir /mnt/data0/Sougata/TTS/ZipVoice_models/Zipvoice/4-epochs/zipvoice_custom \
  --checkpoint-name iter-60000-avg-2.pt \
  --tokenizer simple \
  --prompt-wav  /mnt/data0/Sougata/TTS/zipvoice-docker/backend/vocals_66fbc2709def491d07f3a8dd_0.wav \
  --prompt-text "किन्नी बार कहा, किन्ना बढ़ा उपरेशण, उन हिजूरो के हाते मद दो, तीन साल, तीन साल की प्लानिंग गई पानी में," \
  --text "வணக்கம் ஈரோஸ்" \
  --res-wav-path ../../zipvoice-docker/backend/test.wav

