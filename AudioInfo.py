import librosa
import yt_dlp
import os
import json
import soundfile as sf
import noisereduce as nr
import matplotlib.pyplot as plt
import tkinter
from tkinter import ttk, scrolledtext
from urllib.parse import urlparse
from youtube_transcript_api import YouTubeTranscriptApi

#Hàm hiển thị đoạn văn bản lên scroll view
def show_text(_inputText):
    wrap_content = tkinter.Tk()
    wrap_content.title("Kết quả")
    
    frame = ttk.Frame(wrap_content)
    frame.pack(padx=10, pady=10, fill=tkinter.BOTH, expand=True)

    text_area = scrolledtext.ScrolledText(frame, wrap=tkinter.WORD, width=100, height=30)
    text_area.pack(fill=tkinter.BOTH, expand=True)

    text_area.insert(tkinter.END, _inputText)

    wrap_content.mainloop()

# Kiểm tra url nhập vào có phải là link youtube không?
def isWebLink(_targetUrl):
    try:
        if 'youtube' in _targetUrl.lower():
            resultUrl = urlparse(_targetUrl)
            return all([resultUrl.scheme, resultUrl.netloc])
        else:
            return False 
    except ValueError:
        return False

# Load file text từ đường link youtube thông qua youtube_transcript_api
def getReferenceText(_targetUrl, _langType):
    video_id = _targetUrl.replace('https://www.youtube.com/watch?v=', '')
    try:
        # Lấy transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id,languages=[_langType])
        # Ghép các đoạn text thành một chuỗi hoàn chỉnh
        script = " ".join('' if item['text'].startswith('[') and item['text'].endswith(']') else item['text'] for item in transcript)        
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
def downloadAudioFromYoutubeLink(_targetUrl, _outputFile="audio.mp3"):
    ydl_opts = {
        'format': 'bestaudio/best',  # Choose the best available audio format
        'outtmpl': _outputFile,  # Set the output filename
        'noplaylist': True, # only download single video if it's a playlist
    }
    try:
        if os.path.exists(_outputFile):
            os.remove(_outputFile)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([_targetUrl])
        print(f"Audio downloaded successfully to {_outputFile}")
        return _outputFile 
    except Exception as e:
        print(f"An error occurred: {e}")
        return None 

# Đọc file audio.json
def loadAudioData(_jsonFile="audio.json"):
    if os.path.exists(_jsonFile):
        with open(_jsonFile, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):  # Nếu là list, chuyển sang dạng dictionary
                    return {"data": data}
                return data  # Trả về dictionary nếu đúng format
            except json.JSONDecodeError:
                return {"data": []}
    return {"data": []}

# Ghi dữ liệu mới vào audio.json
def saveScriptAudioData(_jsonData, _jsonFile="audio.json"):
    with open(_jsonFile, "w", encoding="utf-8") as f:
        json.dump(_jsonData, f, ensure_ascii=False, indent=4)

# Kiểm tra xem link đã tồn tại trong dữ liệu chưa
def getAudioScript(_targetUrl, _langType):
    _jsonData = loadAudioData()
    for entry in _jsonData["data"]:
        if entry.get("link").startswith(_targetUrl) or _targetUrl.startswith(entry.get("link")):
            _existingContent = entry["content"]
            if _existingContent == '':
                _existingContent = getReferenceText(_targetUrl, _langType)
                entry["content"] = _existingContent
                saveScriptAudioData(_jsonData)
            return _existingContent
    # Nếu link chưa có, thêm mới
    _content = getReferenceText(_targetUrl, _langType)
    _jsonData["data"].append({"link":_targetUrl, "content": _content})
    saveScriptAudioData(_jsonData)
    print("Dữ liệu mới đã được thêm vào audio.json.")
    return _content

# Bộ lọc nhiễu
def boLocNhieu(_filePath):
    audio, sr = librosa.load(_filePath, sr=16000)
    
    # Lấy một phần tín hiệu làm noise profile (VD: đoạn đầu tiên)
    noise_part = audio[-16000:]

    # Áp dụng bộ lọc giảm nhiễu
    denoised_audio = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_part, prop_decrease=0.5)

    sf.write(_filePath.replace('.mp3', '_1.mp3'), denoised_audio, sr)
     # Vẽ đồ thị
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr, alpha=0.5, color='r')
    plt.title("Tín hiệu gốc (Original Audio)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    librosa.display.waveshow(denoised_audio, sr=sr, alpha=0.5, color='b')
    plt.title("Tín hiệu sau lọc nhiễu (Denoised Audio)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

# Load file text từ đường link youtube thông qua youtube_transcript_api
def getListReferenceText(_defaultLangType='vi'):
    _jsonData = loadAudioData()
    _isNeedSave = False
    for entry in _jsonData["data"]:
        if entry["content"] == '':
            entry["link"] = entry["link"][0:entry["link"].index('&')] if '&' in entry["link"] else entry["link"]
            entry["content"] = getAudioScript(entry["link"], _defaultLangType)
            _isNeedSave = True
    if _isNeedSave: 
        saveScriptAudioData(_jsonData)

def popupInputLinkFileName():
    """Tạo cửa sổ GUI để nhập tên và tuổi."""

    def getInfo():
        global video_url, file_path
        video_url = entryLink.get()
        file_path = entryFileName.get()
        if file_path is None or video_url is None:  # Check if file_path is None
            ket_qua_label.config(text="Tên file và link không được để trống.")
            return
        if not file_path.endswith('.mp3'):
            ket_qua_label.config(text="Tên file phải kết thúc bằng .mp3")
            return
        video_url = video_url[0:video_url.index('&')] if '&' in video_url else video_url
        file_path = downloadAudioFromYoutubeLink(video_url, file_path)
        if file_path == None:
            ket_qua_label.config(text="Link không đúng.")
            return
        ket_qua_label.config(text=f"Link: {video_url}\n Tên file: {file_path}")
        window.destroy()

    window = tkinter.Tk()
    window.title("Nhập thông tin")

    # Nhãn và hộp nhập liệu cho tên
    tkinter.Label(window, text="Đường dẫn liên kết:").pack()
    entryLink = tkinter.Entry(window)
    entryLink.pack()

    # Nhãn và hộp nhập liệu cho tuổi
    tkinter.Label(window, text="Tên file mong muốn:").pack()
    entryFileName = tkinter.Entry(window)
    entryFileName.pack()
    entryFileName.setvar("abc")

    # Nút bấm để lấy thông tin
    button = tkinter.Button(window, text="Lấy thông tin", command=getInfo)
    button.pack()

    # Nhãn để hiển thị kết quả
    ket_qua_label = tkinter.Label(window, text='')
    ket_qua_label.pack()

    window.mainloop()
    return video_url, file_path