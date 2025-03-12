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
import requests
import threading

#Hàm hiển thị đoạn văn bản lên scroll view
def showResultText(_resultText):
    wrap_content = tkinter.Tk()
    wrap_content.title("Kết quả")
    
    frame = ttk.Frame(wrap_content)
    frame.pack(padx=10, pady=10, fill=tkinter.BOTH, expand=True)

    text_area = scrolledtext.ScrolledText(frame, wrap=tkinter.WORD, width=100, height=30)
    text_area.pack(fill=tkinter.BOTH, expand=True)

    text_area.insert(tkinter.END, _resultText)

    wrap_content.mainloop()

# Kiểm tra url nhập vào có phải là link youtube không?
def isYoutubeLink(_targetUrl):
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

def downloadAudioFromYoutubeLink(_targetUrl, _outputFile="audio.mp3", _resultLabel = None):
    # Tạo cửa sổ Tkinter
    window = tkinter.Tk()
    window.title("Downloading Audio")
    window.geometry("400x150")

    # Label hiển thị trạng thái
    statusLabel = tkinter.Label(window, text=f"Downloading: {_outputFile}")
    statusLabel.pack(pady=10)

    # Thanh tiến trình
    progress = ttk.Progressbar(window, length=300, mode='determinate')
    progress.pack(pady=10)

    # Label hiển thị phần trăm
    percentLabel = tkinter.Label(window, text="0%")
    percentLabel.pack(pady=5)

    # Hook để cập nhật tiến trình
    def myHook(d):
        if d['status'] == 'downloading':
            totalBytes = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            downloadedBytes = d.get('downloaded_bytes', 0)
            if totalBytes > 0:
                percentage = (downloadedBytes / totalBytes) * 100
                progress['value'] = percentage
                percentLabel.config(text=f"{percentage:.1f}%")
                window.update_idletasks()
        elif d['status'] == 'finished':
            progress['value'] = 100
            percentLabel.config(text="100% - Converting...")
            window.update_idletasks()
            if not _resultLabel is None:
                _resultLabel.config(text=f"Link: {_videoUrl[:47] + '...' if len(_videoUrl) > 50 else _videoUrl}\n Tên file: {_filePath}")
            

    # Cấu hình yt-dlp
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': _outputFile,
        'noplaylist': True,
        'progress_hooks': [myHook],
    }

    # Hàm tải chạy trong thread riêng
    def downloadThread():
        try:
            if os.path.exists(_outputFile):
                os.remove(_outputFile)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([_targetUrl])
            statusLabel.config(text=f"Completed: {_outputFile}")
            percentLabel.config(text="100% - Done!")
        except Exception as e:
            statusLabel.config(text=f"Error: {e}")
            percentLabel.config(text="Failed")
        finally:
            # Thêm nút đóng cửa sổ khi hoàn tất
            closeButton = tkinter.Button(window, text="Close", command=window.destroy)
            closeButton.pack(pady=10)

    # Chạy tải trong thread để không làm freeze GUI
    threading.Thread(target=downloadThread, daemon=True).start()

    # Chạy vòng lặp Tkinter
    window.mainloop()

    # Trả về đường dẫn file nếu thành công
    if os.path.exists(_outputFile):
        return _outputFile
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
            entry["link"] = entry["link"][0:entry["link"].index('&')] if '&' in entry["link"] and 'youtube' in entry["link"] else entry["link"]
            entry["content"] = getAudioScript(entry["link"], _defaultLangType)
            _isNeedSave = True
    if _isNeedSave: 
        saveScriptAudioData(_jsonData)

def popupInputLinkFileName():
    """Tạo cửa sổ GUI để nhập tên và tuổi."""

    def getInfo():
        global _videoUrl, _filePath
        _videoUrl = entryLink.get()
        _filePath = entryFileName.get()
        if _filePath is None or _videoUrl is None:  # Check if file_path is None
            resultLabel.config(text="Tên file và link không được để trống.")
            return
        if _filePath is None or _filePath == '':
            _filePath = 'default_name.mp3'
        if not _filePath.endswith('.mp3'):
            _filePath = _filePath + '.mp3'
        if isYoutubeLink(_videoUrl) :
            _filePath = downloadAudioFromYoutubeLink(_videoUrl, _filePath, resultLabel)
        else :
            _filePath = downloadFile(_videoUrl, _filePath, resultLabel)

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

    # Nút bấm để lấy thông tin
    button = tkinter.Button(window, text="Lấy thông tin", command=getInfo)
    button.pack()

    # Nhãn để hiển thị kết quả
    resultLabel = tkinter.Label(window, text='')
    resultLabel.pack()

    window.mainloop()
    return _videoUrl, _filePath


def downloadFile(_url, _outputFile = 'audio.mp3', _resultLabel = None):
    # Tạo cửa sổ Tkinter
    root = tkinter.Tk()
    root.title("File Download")
    root.geometry("400x150")

    # Label hiển thị tên file
    statusLabel = tkinter.Label(root, text=f"Downloading: {_outputFile}")
    statusLabel.pack(pady=10)

    # Thanh tiến trình
    progress = ttk.Progressbar(root, length=300, mode='determinate')
    progress.pack(pady=10)

    # Label hiển thị phần trăm
    percent_label = tkinter.Label(root, text="0%")
    percent_label.pack(pady=5)

    # Hàm tải file chạy trong thread riêng
    def downloadThread():
        try:
            if os.path.exists(_outputFile):
                os.remove(_outputFile)
            # Gửi yêu cầu với stream=True để tải theo từng phần
            response = requests.get(_url, stream=True)
            totalSize = int(response.headers.get('content-length', 0))

            if response.status_code == 200:
                # Ghi file và cập nhật tiến trình
                with open(_outputFile, 'wb') as file:
                    downloaded = 0
                    chunkSize = 1024
                    for data in response.iter_content(chunk_size=chunkSize):
                        size = file.write(data)
                        downloaded += size
                        if totalSize > 0:
                            percentage = (downloaded / totalSize) * 100
                            progress['value'] = percentage
                            percent_label.config(text=f"{percentage:.1f}%")
                            root.update_idletasks()
                
                statusLabel.config(text=f"Completed: {_outputFile}")
                percent_label.config(text="100% - Done!")
                if not _resultLabel is None:
                    _resultLabel.config(text=f"Link: {_videoUrl[:47] + '...' if len(_videoUrl) > 50 else _videoUrl}\n Tên file: {_outputFile}")
            else:
                statusLabel.config(text=f"Error: HTTP {response.status_code}")
                percent_label.config(text="Failed")

        except Exception as e:
            statusLabel.config(text=f"Error: {e}")
            percent_label.config(text="Failed")

        finally:
            # Thêm nút đóng
            closeButton = tkinter.Button(root, text="Close", command=root.destroy)
            closeButton.pack(pady=10)

    # Chạy tải trong thread riêng để không làm freeze GUI
    threading.Thread(target=downloadThread, daemon=True).start()

    # Chạy vòng lặp Tkinter
    root.mainloop()

    # Kiểm tra xem file đã tải thành công chưa
    if os.path.exists(_outputFile):
        print(f"Đã tải xong: {_outputFile}")
        return _outputFile
    return None