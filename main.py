import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from time import time
import argparse
import sys

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)


from cv2_enumerate_cameras import enumerate_cameras


def read_image(path):
    image = cv2.imread(path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image

def detect_faces_bboxes(image):
    # Детекция лиц
    results = face_detection.process(image)

    bboxes = []
    if results.detections:
        for detection in results.detections:

            image_height, image_width, _ = image.shape

            # Извлечение координат ограничивающей рамки
            bbox = detection.location_data.relative_bounding_box
            x_min = int(bbox.xmin * image_width)
            y_min = int(bbox.ymin * image_height)
            box_width = int(bbox.width * image_width)
            box_height = int(bbox.height * image_height)
            box_width *= 1.8
            box_height *= 1.8
            x_min -= int(box_width * 0.2)
            y_min -= int(box_height * 0.3)
            x_min = max(0, x_min)
            y_min = max(0, y_min)

            bboxes.append((x_min, y_min, x_min + int(box_width), y_min + int(box_height)))

    return bboxes



def get_faces_images(image, bboxes):
    
    face_images = []

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox

        face_images.append(image[y_min:y_max, x_min:x_max])

    return face_images


def prepare_batch(images: list):

    images_np = np.stack(images)

    B, H, W, C = images_np.shape

    mean = np.array([0.496, 0.502, 0.504], dtype=np.float32)
    std = np.array([0.254, 0.2552, 0.2508], dtype=np.float32)

    #normalize
    if images_np.dtype == np.uint8:
        images_np = images_np.astype(np.float32) / 255.0

    resized_batch = np.zeros((B, 256, 256, C), dtype=np.float32)
    
    for i in range(B):
        resized_batch[i] = cv2.resize(images_np[i], (256, 256))  # (W, H)

    # Перевод в формат (B, C, H, W)
    transposed = np.transpose(resized_batch, (0, 3, 1, 2))  # (B, C, H, W)

    # Нормализация по каналам
    for i in range(transposed.shape[1]):  # для каждого канала
        transposed[:, i, :, :] = (transposed[:, i, :, :] - mean[i]) / std[i]

    contiguous_transposed = np.ascontiguousarray(transposed)


    return contiguous_transposed.astype(np.float32)

def facepoints(face_tensors, session):

    input_name = session.get_inputs()[0].name

    predictions = session.run(None, {input_name: face_tensors})

    facepoints = predictions[0]

    return facepoints


def draw_on_face(images, facepoints):
    for image, points in zip(images, facepoints):

        original_h, original_w = image.shape[:2]

        landmarks_x = points[::2]
        landmarks_y = points[1::2]

        # # Восстановление координат в исходном разрешении
        landmarks_x *= original_h
        landmarks_y *= original_w

        for x, y in zip(landmarks_x, landmarks_y):
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        cv2.imwrite(f'_.jpg', image)


def draw_keypoints(image, bboxes, facepoints):
    
    img = image.copy()

    for bbox, points in zip(bboxes, facepoints):
        x_min, y_min, x_max, y_max = bbox

        landmarks_x = points[::2]
        landmarks_y = points[1::2]


        w = x_max - x_min
        h = y_max - y_min

        landmarks_x *= h
        landmarks_y *= w

        for x, y in zip(landmarks_x, landmarks_y):
            cv2.circle(img, (x_min + int(x), y_min + int(y)), 2, (0, 255, 0), -1)

    return img

    


def process_image(image: np.ndarray, session) -> np.ndarray:

    image = cv2.flip(image, 1)

    bboxes = detect_faces_bboxes(image)

    face_images = get_faces_images(image, bboxes)

    if not face_images:
        return image

    try:
        face_tensors = prepare_batch(face_images)

        points = facepoints(face_tensors, session)
    except Exception as e:
        return image

    return draw_keypoints(image, bboxes, points)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ONNX demo')
    parser.add_argument('-cam', type=int, help='camera index', required=False)



    onnx_model = "mobilenet_relu.onnx"
    sess_options = ort.SessionOptions()

    session = ort.InferenceSession(
        onnx_model, providers=["CPUExecutionProvider"], sess_options=sess_options
    )

    args = parser.parse_args()

    print("available cameras: ")
    cams = set()
    for camera_info in enumerate_cameras(cv2.CAP_MSMF):
        print(f'{camera_info.index}: {camera_info.name} {camera_info.path}')
        cams.add(camera_info.index)

    if args.cam is not None:
        if args.cam not in cams:
            print(f'Camera with index {args.cam} unavalilable')
            sys.exit()
        print(f"Open camera with index: {args.cam}")
        cap = cv2.VideoCapture(args.cam)
    else:
        print("open default cam")
        cap = cv2.VideoCapture(0)

    # Проверка открытия камеры
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру.")
        exit()

    # print("Нажмите 'Esc' для выхода.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: Не удалось получить кадр.")
            break
        
        start = time()
        image_wit_keypoints = process_image(frame, session)

        print(f"{(time() - start) * 1000:.2f} ms")

        # Отображение изображения
        cv2.imshow('Face detector', image_wit_keypoints)

        # Выход по нажатию 'Esc'
        if cv2.waitKey(20) == 27:
            break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
