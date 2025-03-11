import torch
import whisper
import librosa
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd
import jiwer  
import yt_dlp
import os
import sys
import tkinter
from tkinter import ttk, scrolledtext
from urllib.parse import urlparse
from datasets import load_dataset, Audio
from youtube_transcript_api import YouTubeTranscriptApi
import json

#Hàm hiển thị đoạn văn bản lên scroll view
def show_text(input_text):
    wrap_content = tkinter.Tk()
    wrap_content.title("Kết quả")

    frame = ttk.Frame(wrap_content)
    frame.pack(padx=10, pady=10, fill=tkinter.BOTH, expand=True)

    text_area = scrolledtext.ScrolledText(frame, wrap=tkinter.WORD, width=100, height=30)
    text_area.pack(fill=tkinter.BOTH, expand=True)

    text_area.insert(tkinter.END, input_text)

    wrap_content.mainloop()

# Kiểm tra url nhập vào có phải là link youtube không?
def isWebLink(url):
    try:
        if 'youtube' in param.lower():
            resultUrl = urlparse(url)
            return all([resultUrl.scheme, resultUrl.netloc])
        else:
            return False 
    except ValueError:
        return False

# Load file text từ đường link youtube thông qua youtube_transcript_api
def get_reference_text(_targetUrl, _langType):
    video_id = _targetUrl.replace('https://www.youtube.com/watch?v=', '')
    try:
        # Lấy transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id,languages=[_langType])
        # Ghép các đoạn text thành một chuỗi hoàn chỉnh
        script = " ".join(item['text'].replace('[', '').replace(']', '') for item in transcript)
        for item in transcript:
            #  print(f"{item['text']} (Thời gian: {item['start']}s)")
            return script
    except Exception as e:
        if "disabled" in str(e).lower():
            print("Video này không có transcript hoặc chủ video tắt captions.")
        elif "not found" in str(e).lower():
            print("Không tìm thấy transcript, có thể video không hỗ trợ.")
        else:
            print(f"Lỗi khác: {e}")
        return ''


# Tải file audio từ youtube link
def download_audio_from_youtube(video_url, output_filename="audio.mp3"):
    ydl_opts = {
        'format': 'bestaudio/best',  # Choose the best available audio format
        'outtmpl': output_filename,  # Set the output filename
        'noplaylist': True, # only download single video if it's a playlist
    }
    try:
        if os.path.exists(output_filename):
            os.remove(output_filename)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        print(f"Audio downloaded successfully to {output_filename}")
        return output_filename 
    except Exception as e:
        print(f"An error occurred: {e}")
        return None 

video_url = 'https://www.youtube.com/watch?v=WmwiO-hfoyE'   # Link mặc định
file_path = 'Youtube.mp3'                                   # Tên file mặc định

for index, param in enumerate(sys.argv):
   if index > 0:
        if isWebLink(param):
            video_url = param
        elif param.lower().endswith('.mp3') & (param == 'Youtube.mp3'): # Lưu file với format *.mp3
            file_path = param

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_t = whisper.load_model("small").to(device)

file_path = download_audio_from_youtube(video_url, file_path)

arrContent = []
try:
    audio_data = whisper.load_audio(file_path)
    result = model_t.transcribe(file_path)
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
ground_truth_text = get_reference_text(video_url, detected_language)

# Giải mã âm thanh
# options = whisper.DecodingOptions(fp16=False)
# result = whisper.decode(model_t, mel, options)

transformation = jiwer.Compose([
    jiwer.RemovePunctuation(),   # Xóa dấu câu
    jiwer.RemoveWhiteSpace(replace_by_space=True),  # Chuẩn hóa khoảng trắng
    jiwer.ToLowerCase(),         # Chuyển về chữ thường
    jiwer.RemoveMultipleSpaces(),  # Xóa khoảng trắng thừa
])
# Đọc file audio.json
def load_audio_data(json_file="audio.json"):
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):  # Nếu là list, chuyển sang dạng dictionary
                    return {"data": data}
                return data  # Trả về dictionary nếu đúng format
            except json.JSONDecodeError:
                return {"data": []}
    return {"data": []}

# Ghi dữ liệu mới vào audio.json
def save_audio_data(data, json_file="audio.json"):
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Kiểm tra xem link đã tồn tại trong dữ liệu chưa
def check_and_update_audio_data(video_url, ground_truth_text):
    data = load_audio_data()
    for entry in data["data"]:
        if entry.get("link") == video_url:
            existing_content = entry["content"]
            return existing_content
        # Nếu link chưa có, thêm mới
    data["data"].append({"link": video_url, "content": ground_truth_text})
    save_audio_data(data)
    print("Dữ liệu mới đã được thêm vào audio.json.")
    return ground_truth_text



transcription = result["text"]

ground_final = check_and_update_audio_data(video_url, ground_truth_text)


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
show_text(text_to_show)