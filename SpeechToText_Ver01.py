import torch
import whisper
import librosa
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd
import jiwer  
import AudioInfo

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_t = whisper.load_model("small").to(device)

video_url, file_path = AudioInfo.popupInputLinkFileName()
video_url = video_url[0:video_url.index('&')] if '&' in video_url and 'youtube' in video_url else video_url

try:
    originalAudio, denoisedAudio = AudioInfo.boLocNhieu(file_path)
    resultOriAud = model_t.transcribe(file_path, fp16 = True if device == 'cuda' else False )
    print("Audio loaded successfully!")
    resultDeNAud = model_t.transcribe(file_path.replace('.mp3', '_1.mp3'), fp16 = True if device == 'cuda' else False )
    print("Audio with filter loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure FFmpeg is installed and added to your PATH.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# # Vẽ biểu đồ
originalAudioTrim = whisper.pad_or_trim(originalAudio)
melOriginalAudio = whisper.log_mel_spectrogram(originalAudioTrim).to(model_t.device)
AudioInfo.showGraphCompairMelSpec(originalAudio, denoisedAudio, melOriginalAudio)

# Chuyển mảng thành chuỗi
originalAudioStr = np.array2string(originalAudio, separator=', ')

sr=22050
ipd.Audio(originalAudioTrim, rate = sr)
_, probs = model_t.detect_language(melOriginalAudio)
detectedLanguage = max(probs, key = probs.get)

transformation = jiwer.Compose([
    jiwer.RemovePunctuation(),                      # Xóa dấu câu
    jiwer.RemoveWhiteSpace(replace_by_space=True),  # Chuẩn hóa khoảng trắng
    jiwer.ToLowerCase(),                            # Chuyển về chữ thường
    jiwer.RemoveMultipleSpaces(),                   # Xóa khoảng trắng thừa
])

transcriptionOriAud = resultOriAud["text"]
transcriptionDeNAud = resultDeNAud["text"]

transClean = transformation(transcriptionOriAud)
transCleanDe = transformation(transcriptionDeNAud)

# Tính lại và hiển thị văn bản
if 'youtube' in video_url:
    ground_final = AudioInfo.getAudioScript(video_url, detectedLanguage)
    gtClean = transformation(ground_final)
    werScore = jiwer.wer(gt_clean, transClean)
    werScoreDe = jiwer.wer(gt_clean, transCleanDe)

    # Đoạn văn bản cần hiển thị
    text_to_show =  f"Dữ liệu âm thanh      : {originalAudioStr}\n\n" \
                    f"Kết quả nhận chưa lọc : {transClean}\n\n" \
                    f"Kết quả nhận đã lọc   : {transCleanDe}\n\n" \
                    f"Kết quả gốc           : {gtClean}\n\n" \
                    f"Loại ngôn ngữ         : {detectedLanguage}\n\n" \
                    f"Word Error Rate (WER) 1: {werScore:.2%}\n\n" \
                    f"Word Error Rate (WER) 2: {werScoreDe:.2%}\n\n" \
    
else :
    # Đoạn văn bản cần hiển thị
    text_to_show =  f"Dữ liệu âm thanh      : {originalAudioStr}\n\n" \
                    f"Kết quả nhận chưa lọc : {transClean}\n\n" \
                    f"Kết quả nhận đã lọc   : {transCleanDe}\n\n" \
                    f"Loại ngôn ngữ         : {detectedLanguage}\n\n"

# Gọi hàm tạo cửa sổ và hiển thị văn bản
AudioInfo.showResultText(text_to_show)