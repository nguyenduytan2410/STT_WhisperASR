import torch
import whisper
import sys
import numpy as np
import IPython.display as ipd
import jiwer  
import AudioInfo
from transformers import WhisperForConditionalGeneration, WhisperProcessor

video_url, file_path, strModel = AudioInfo.popupInputLinkFileName()

print(strModel)

if video_url is None or file_path is None or strModel is None:
    sys.exit(1)    # Dừng chương trình

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_t = whisper.load_model(strModel).to(device)

# Load processor và model bằng Transformers
processor = WhisperProcessor.from_pretrained("openai/whisper-small")  # Thay "openai/whisper-small" bằng model gốc bạn đã dùng để fine-tune

# (Tùy chọn) Lưu processor vào thư mục checkpoint để dùng sau
processor.save_pretrained('whisper_finetune/checkpoint-187')

# Load model từ thư mục checkpoint
model = WhisperForConditionalGeneration.from_pretrained('whisper_finetune/checkpoint-187', local_files_only=True)

# Chuyển model sang thiết bị
model = model.to(device)
model.eval()

# Kết hợp với Whisper để chuyển đổi âm thanh thành văn bản
whisper_model = whisper.load_model("small")  # Load một model cơ bản để lấy pipeline
whisper_model.model = model  # Thay thế model bằng model đã fine-tune

video_url = video_url[0:video_url.index('&')] if '&' in video_url and 'youtube' in video_url else video_url

try:
    _isFP16 = True if device == 'cuda' else False
    originalAudio, denoisedAudio = AudioInfo.boLocNhieu(file_path)
    print("Noise-free audio created successfully!")
    resultOriAud = model_t.transcribe(file_path, fp16 = _isFP16)
    print("Audio loaded successfully!")
    resultOriAud_reTrain = whisper_model.transcribe(file_path, fp16 = _isFP16)
    print("Audio retrain loaded successfully!")
    resultDeNAud = model_t.transcribe(file_path.replace('.mp3', '_denoise.mp3'), fp16 = _isFP16)
    print("Audio with filter loaded successfully!")
    resultDeNAud_reTrain = whisper_model.transcribe(file_path.replace('.mp3', '_denoise.mp3'), fp16 = _isFP16)
    print("Audio with filter retrain loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure FFmpeg is installed and added to your PATH.")
    sys.exit(1)    # Dừng chương trình
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)    # Dừng chương trình

# # Vẽ biểu đồ
originalAudioTrim = whisper.pad_or_trim(originalAudio)
denoisedAudioTrim = whisper.pad_or_trim(denoisedAudio)
melOriginalAudio = whisper.log_mel_spectrogram(originalAudioTrim, n_mels = 128 if strModel == 'large' or strModel == 'turbo' else 80).to(model_t.device)
melDeAudio = whisper.log_mel_spectrogram(denoisedAudioTrim, n_mels = 128 if strModel == 'large' or strModel == 'turbo' else 80).to(model_t.device)
AudioInfo.showGraphCompairMelSpec(originalAudio, denoisedAudio, melOriginalAudio, melDeAudio)

# Chuyển mảng thành chuỗi
originalAudioStr = np.array2string(originalAudio, separator=', ')

sr = 16000
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
transcriptionOriAud_reTrain = resultOriAud_reTrain["text"]
transcriptionDeNAud = resultDeNAud["text"]
transcriptionDeNAud_reTrain = resultDeNAud_reTrain["text"]

transClean = transformation(transcriptionOriAud)
transClean_reT = transformation(transcriptionOriAud_reTrain)
transCleanDe = transformation(transcriptionDeNAud)
transCleanDe_reT = transformation(transcriptionDeNAud_reTrain)



text_to_show =  f"Dữ liệu âm thanh      : {originalAudioStr}\n\n" \
                f"Kết quả nhận chưa lọc1: {transClean}\n\n" \
                f"Kết quả nhận chưa lọc2: {transClean_reT}\n\n" \
                f"Kết quả nhận đã lọc 1 : {transCleanDe}\n\n" \
                f"Kết quả nhận đã lọc 2 : {transCleanDe_reT}\n\n" \
                f"Loại ngôn ngữ         : {detectedLanguage}\n\n"

# Tính lại và hiển thị văn bản
if 'youtube' in video_url:
    referenceText = AudioInfo.getAudioScript(video_url, detectedLanguage)
    if referenceText != '':
        snr = AudioInfo.snr(originalAudio, denoisedAudio)
        gtClean = transformation(referenceText)
        werScore = jiwer.wer(gtClean, transClean)
        werScore2 = jiwer.wer(gtClean, transClean_reT)
        werScoreDe = jiwer.wer(gtClean, transCleanDe)
        werScoreDe2 = jiwer.wer(gtClean, transCleanDe_reT)

        # Đoạn văn bản cần hiển thị
        text_to_show += f"Dữ liệu làm tham chiếu : {gtClean}\n\n" \
                        f"SNR của bộ lọc         : {snr:.2f} dB\n\n" \
                        f"Word Error Rate (WER) 1: {werScore:.2%}\n\n" \
                        f"Word Error Rate (WER) 2: {werScore2:.2%}\n\n" \
                        f"Word Error Rate (WER) 3: {werScoreDe:.2%}\n\n" \
                        f"Word Error Rate (WER) 4: {werScoreDe2:.2%}\n\n" \

# Gọi hàm tạo cửa sổ và hiển thị văn bản
AudioInfo.showResultText(text_to_show)