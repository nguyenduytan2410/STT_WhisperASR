import torch
import whisper
import librosa
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd
import jiwer  
import sys
import AudioInfo

# video_url = 'https://www.youtube.com/watch?v=Clpcxh9lNTs&ab_channel=SpriteVietnam'   # Link mặc định
# file_path = 'Youtube.mp3'                                   # Tên file mặc định

# for index, param in enumerate(sys.argv):
#    print(param)
#    if index > 0:
#         if AudioInfo.isWebLink(param):
#             video_url = param
#         elif param.lower().endswith('.mp3') and param != '.mp3': # Lưu file với format *.mp3
#             print(param)
#             file_path = param

# print(file_path)
# video_url = video_url[0:video_url.index('&')] if '&' in video_url else video_url

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_t = whisper.load_model("tiny").to(device)

video_url, file_path = AudioInfo.popupInputLinkFileName()
video_url = video_url[0:video_url.index('&')] if '&' in video_url else video_url

arrContent = []
try:
    audio_data = whisper.load_audio(file_path)
    result = model_t.transcribe(file_path, fp16 = True if device == 'cuda' else False )
    print("Audio loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure FFmpeg is installed and added to your PATH.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Chuyển mảng thành chuỗi
audio_str = np.array2string(audio_data, separator=', ')

#lấy độ dài của file âm thanh
try:
    duration = librosa.get_duration(path=file_path)
    print(f"Duration: {duration} seconds")
except Exception as e:
    print(f"File error: {e}")

n_samples = audio_data.shape[0]
delta = duration / n_samples
Fs = 1 / delta
time = np.linspace(0, (n_samples - 1) * delta, n_samples)

# Vẽ tín hiệu âm thanh
plt.figure(figsize = (10, 4))
plt.plot(time, audio_data)
plt.title('Signal')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.show()

# Xử lý và vẽ lại tín hiệu sau khi pad/truncate
audio = whisper.pad_or_trim(audio_data)
n_samples=audio.shape[-1]
time=np.linspace(0,(n_samples-1)*delta,n_samples)

plt.figure(figsize=(10, 4))
plt.plot(time, audio)
plt.title('Trimmed Signal')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.show()

# Tạo Mel Spectrogram
mel = whisper.log_mel_spectrogram(audio).to(model_t.device)

fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))
fig.tight_layout(pad=5.0)

ax1.plot(time, audio)
ax1.set_title('Signal')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Amplitude')

ax2.imshow(mel.cpu().numpy(), interpolation='nearest', aspect='auto')
ax2.set_title('Mel Spectrogram of a Signal')
ax2.set_xlabel('Time (frames)')
ax2.set_ylabel('Mel Scale')
plt.show()

# Phát hiện loại ngôn ngữ
sr=22050
ipd.Audio(audio, rate=sr)
_, probs = model_t.detect_language(mel)
detected_language = max(probs, key=probs.get)

# Giải mã âm thanh
# options = whisper.DecodingOptions(fp16=False)
# result = whisper.decode(model_t, mel, options)

transformation = jiwer.Compose([
    jiwer.RemovePunctuation(),   # Xóa dấu câu
    jiwer.RemoveWhiteSpace(replace_by_space=True),  # Chuẩn hóa khoảng trắng
    jiwer.ToLowerCase(),         # Chuyển về chữ thường
    jiwer.RemoveMultipleSpaces(),  # Xóa khoảng trắng thừa
])

transcription = result["text"]

ground_final = AudioInfo.getAudioScript(video_url, detected_language)


gt_clean = transformation(ground_final)
trans_clean = transformation(transcription)


    #Tính lại WER
wer_score = jiwer.wer(gt_clean, trans_clean)

# Đoạn văn bản cần hiển thị
text_to_show =  f"Dữ liệu âm thanh: {audio_str}\n\n" \
                f"Kết quả nhận: {trans_clean}\n\n" \
                f"Kết quả gốc: {gt_clean}\n\n" \
                f"Loại ngôn ngữ: {detected_language}\n\n" \
                f"Word Error Rate (WER): {wer_score:.2%}\n\n"

# Gọi hàm tạo cửa sổ và hiển thị văn bản
AudioInfo.show_text(text_to_show)