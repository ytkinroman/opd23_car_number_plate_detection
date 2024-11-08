import torch
import ultralytics
import easyocr


class RecognizePlate:
    """
    Класс для распознавания автомобильных номеров на изображении с использованием моделей
    YOLO для обнаружения номерных знаков и EasyOCR для их распознавания.
    """

    def __init__(self):
        """
        Инициализирует объект RecognizePlate с пустыми моделями для обнаружения
        и распознавания текста, а также указывает устройство выполнения ('cpu').
        """
        self.__model_detection = None
        self.__model_recognize = None
        self.device = 'cpu'

    @staticmethod
    def __match_pattern(string: str) -> bool:
        """
        Проверяет, соответствует ли распознанный текст шаблону автомобильного номера.

        :param string: распознанный текст
        :return: bool — True, если текст соответствует шаблону, иначе False
        """
        nums = '0123456789ОOоo'
        alf_eng = 'ABEKHMOPCTYX'
        alf_rus = 'АВЕКНМОРСТУХ'
        alf = alf_rus.lower() + alf_rus + alf_eng.lower() + alf_eng
        mask = [alf, nums, nums, nums, alf, alf]
        flag = True

        if len(string) < 6 or len(string) > 9:
            return False

        for i in range(len(string)):
            if string[i] in mask[i]:
                continue
            else:
                flag = False

        return flag

    def load_detection_model(self, path_2_model: str) -> None:
        """
        Загружает модель YOLO для обнаружения номерных знаков на изображении.

        :param path_2_model: путь к файлу модели YOLO
        """
        self.__model_detection = ultralytics.YOLO(path_2_model)

    def load_recognize_model(self, path_2_model: str) -> None:
        """
        Загружает модель EasyOCR для распознавания текста на номерных знаках.

        :param path_2_model: путь к модели EasyOCR (необязательно)
        """
        self.__model_recognize = easyocr.Reader(['ru'])

    def recognize(self, image: 'numpy.ndarray') -> str:
        """
        Основной метод для распознавания номера на изображении.

        :param image: изображение, считанное с помощью cv2.imread
        :return: строка с распознанным номером или None, если номер не распознан
        """
        return self.__recognize_text(image, self.__detection_plate(image))

    def __detection_plate(self, image) -> 'ultralytics.engine.results.Boxes':
        """
        Обнаруживает номерные знаки на изображении с помощью модели YOLO.

        :param image: изображение
        :return: объект с координатами обнаруженных объектов на изображении
        """
        return self.__model_detection(image, device=self.device)

    def __recognize_text(self, frame, results_detection) -> str:
        """
        Распознает текст на выделенной модели YOLO области номерного знака.

        :param frame: исходное изображение
        :param results_detection: результаты детекции номерных знаков YOLO
        :return: строка с распознанным номером или None, если номер не распознан
        """
        for box in results_detection[0].boxes:  # Результаты для первого изображения
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cropped = frame[int(y1):int(y2), int(x1):int(x2)]
            # Применяем OCR
            results = self.__model_recognize.readtext(cropped)
            # Проверка каждого распознанного текста
            for (bbox, text, prob) in results:
                string = ''.join(text.split())
                if self.__match_pattern(string):
                    return string  # Возвращаем первый совпавший номер
