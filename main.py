import cv2
from RecognitionSystem import RecognizePlate

recog = RecognizePlate()
recog.load_detection_model('best.pt')
recog.load_recognize_model('')

# Указываем путь к видеофайлу
video_path = 'your_file.mp4'

# Открываем видеофайл
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Не удалось открыть видео.")
    exit()

# Получаем параметры исходного видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Настраиваем VideoWriter для сохранения видео
output_video_path = 'path/for/save/processed_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0  # Счётчик кадров
print('Начали обрабюотку видео')
while True:
    # Читаем следующий кадр
    ret, frame = cap.read()

    if not ret:
        print("Обработка видео завершена.")
        break

    frame_count += 1

    # Обрабатываем кадр
    recog.recognize(frame)

    # Сохраняем обработанный кадр в видеофайл
    out.write(frame)

# Освобождаем ресурсы
cap.release()
out.release()
print(f"Видео сохранено: {output_video_path}")