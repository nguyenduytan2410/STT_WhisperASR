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
import whisper

# Xóa kí tự đặc biệt khi đặt tên file
def removeSpecialChars(_inputText):
    # Tạo bảng ánh xạ để xóa mọi ký tự không phải a-z, A-Z, 0-9
    allowed = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
    mapping = str.maketrans('', '', ''.join(c for c in map(chr, range(128)) if c not in allowed))
    return _inputText.translate(mapping)

# Hàm hiển thị đoạn văn bản lên scroll view
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

# Tải file text từ đường link youtube thông qua youtube_transcript_api
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

# Tải file mp3 từ youtube link
def downloadAudioFromYoutubeLink(_targetUrl, _outputFile="audio.mp3", _showError = {}, _showSuccess = {}):
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
            _showSuccess()
            
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
            _showError()
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

# Tải file nhạc mp3 từ link bất kì
def downloadFile(_url, _outputFile = 'audio.mp3',  _showError = {}, _showSuccess = {}):
    # Tạo cửa sổ Tkinter
    window = tkinter.Tk()
    window.title("File Download")
    window.geometry("400x150")

    # Label hiển thị tên file
    statusLabel = tkinter.Label(window, text=f"Downloading: {_outputFile}")
    statusLabel.pack(pady = 10)

    # Thanh tiến trình
    progress = ttk.Progressbar(window, length=300, mode='determinate')
    progress.pack(pady = 10)

    # Label hiển thị phần trăm
    percent_label = tkinter.Label(window, text = "0%")
    percent_label.pack(pady = 5)

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
                            window.update_idletasks()
                
                statusLabel.config(text=f"Completed: {_outputFile}")
                percent_label.config(text="100% - Done!")
                _showSuccess()
            else:
                statusLabel.config(text=f"Error: HTTP {response.status_code}")
                percent_label.config(text="Failed")

        except Exception as e:
            statusLabel.config(text=f"Error: {e}")
            percent_label.config(text="Failed")
            _showError()

        finally:
            # Thêm nút đóng
            closeButton = tkinter.Button(window, text="Close", command = window.destroy)
            closeButton.pack(pady = 10)

    # Chạy tải trong thread riêng để không làm freeze GUI
    threading.Thread(target=downloadThread, daemon=True).start()

    # Chạy vòng lặp Tkinter
    window.mainloop()

    # Kiểm tra xem file đã tải thành công chưa
    if os.path.exists(_outputFile):
        print(f"Đã tải xong: {_outputFile}")
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
def boLocNhieu(_filePath, _sr=22050):
    _originalAudio, _sr = librosa.load(_filePath, sr=_sr)
    
    # Lấy một phần tín hiệu làm noise profile (VD: đoạn đầu tiên)
    _noisePart = _originalAudio[-_sr:]

    # Áp dụng bộ lọc giảm nhiễu
    _denoisedAudio = nr.reduce_noise(y = _originalAudio, sr = _sr, y_noise = _noisePart, prop_decrease = 0.5)
    _filePathDN = _filePath.replace('.mp3', '_1.mp3')
    if os.path.exists(_filePathDN):
        os.remove(_filePathDN)

    sf.write(_filePathDN, _denoisedAudio, _sr)
    return _originalAudio, _denoisedAudio

#Vẽ biểu đồ so sánh và đồ thị Mel Spectrogram của âm thanh gốc
def showGraphCompairMelSpec(_originalAudio, _denoisedAudio, mel, _sr=22050):
     # Vẽ đồ thị
    plt.subplot(2, 2, 1)  # 2 hàng, 2 cột, vị trí 1
    librosa.display.waveshow(_originalAudio, sr = _sr, alpha=0.5, color='r')
    plt.title("Tín hiệu gốc (Original Audio)")
    plt.xlabel("Thời gian (giây)")
    plt.ylabel("Biên độ")

    # Biểu đồ 2: Tín hiệu sau lọc nhiễu
    plt.subplot(2, 2, 2)  # 2 hàng, 2 cột, vị trí 2
    librosa.display.waveshow(_denoisedAudio, sr=_sr, alpha=0.5, color='b')
    plt.title("Tín hiệu sau lọc nhiễu (Denoised Audio)")
    plt.xlabel("Thời gian (giây)")
    plt.ylabel("Biên độ")

    # Biểu đồ 3: Mel Spectrogram (chiếm cả hai cột ở hàng dưới
    plt.subplot(2, 1, 2)  # 2 hàng, 1 cột, vị trí 2 (hàng dưới)
    plt.imshow(mel.cpu().numpy(), interpolation='nearest', aspect='auto')
    plt.title("Mel Spectrogram của tín hiệu")
    plt.xlabel("Thời gian (khung)")
    plt.ylabel("Thang Mel")
    plt.colorbar(label='Cường độ (dB)')  # Thêm thanh màu để dễ đọc

    # Điều chỉnh khoảng cách giữa các biểu đồ
    plt.tight_layout()

    # Hiển thị
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

# Mở cửa sổ để nhập thông tin link và tên file lưu vào bộ nhớ
def popupInputLinkFileName():
    def getInfo():
        global _videoUrl, _filePath
        _videoUrl = entryLink.get()
        _filePath = entryFileName.get()
        if _videoUrl is None or _videoUrl == '':  # Check if file_path is None
            resultLabel.config(text="Link không được để trống.")
            return
        if _filePath is None or _filePath == '':
            _filePath = 'default_name'
            entryFileName.insert(0, _filePath) 
        _filePath = _filePath[0:_filePath.index('.mp3')] if '.mp3' in _filePath else _filePath
        _filePath = removeSpecialChars(_filePath) + '.mp3'
        if isYoutubeLink(_videoUrl) :
            _filePath = downloadAudioFromYoutubeLink(_videoUrl, _filePath, showError, showSuccess)
        else :
            _filePath = downloadFile(_videoUrl, _filePath, showError, showSuccess)
    
    def showError():
        resultLabel.config(text="Link không hợp lệ!")
        buttonGet.pack_forget()
        buttonClose.pack_forget()
        buttonGet.pack()
        buttonGet.config(text="Tải lại file")
    def showSuccess():
        resultLabel.config(text=f"Link: {_videoUrl[:47] + '...' if len(_videoUrl) > 50 else _videoUrl}\n Tên file: {_filePath}")
        buttonGet.pack_forget()
        buttonClose.pack_forget()
        buttonGet.pack(side=tkinter.LEFT, padx=10)
        buttonClose.pack(side=tkinter.LEFT, padx=10)
        buttonGet.config(text="Tải lại file")

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

    # Tạo Frame để chứa hai nút
    buttonFrame = tkinter.Frame(window)
    buttonFrame.pack(pady=10)

    # Nhãn để hiển thị kết quả
    resultLabel = tkinter.Label(window, text='')
    resultLabel.pack()

    # Nút bấm để lấy thông tin
    buttonGet = tkinter.Button(buttonFrame, text="Tải file", command = getInfo)
    buttonGet.pack()

    # Nút bấm để vẽ biểu đồ
    buttonClose = tkinter.Button(buttonFrame, text="Đóng", command = window.destroy)

    window.mainloop()
    return _videoUrl, _filePath