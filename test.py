import glob
import os
import cv2
import time
from numpy import array as np_array
import face_detection

TARGET_WIDTH = 640

def resize_image(image):
    original_width = image.shape[1]
    original_height = image.shape[0]
    aspect_ratio = original_width / original_height
    target_height = int(TARGET_WIDTH / aspect_ratio)
    resized_image = cv2.resize(image, (TARGET_WIDTH, target_height), interpolation=cv2.INTER_AREA)
    padded_resized_image = cv2.copyMakeBorder(resized_image, 0, TARGET_WIDTH - target_height, 0, 0, cv2.BORDER_CONSTANT, value=0)

    return padded_resized_image

def draw_faces(im, bboxes):
    h, w = im.shape[0], im.shape[1]
    for bbox in bboxes:
        x0, y1, x1, y1 = bbox * np_array([w, h, w, h])
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)


if __name__ == "__main__":
    impaths = "images"
    impaths = glob.glob(os.path.join(impaths, "*.jpg"))
    detector = face_detection.build_detector(
        "DSFDDetectorTensorRT",
        max_resolution=1080,
    )
    for impath in impaths:
        if impath.endswith("out.jpg"): continue
        im = cv2.imread(impath)
        resized_im = resize_image(im)
        # im = cv2.resize(im, (300, 300), interpolation = cv2.INTER_AREA)
        print("Processing:", impath)
        t = time.time()
        dets = detector.detect(
            im[:, :, ::-1]
        )[:, :4]
        dets /= TARGET_WIDTH
        print(f"Detection time: {time.time()- t:.3f}")
        draw_faces(im, dets)
        imname = os.path.basename(impath).split(".")[0]
        output_path = os.path.join(
            os.path.dirname(impath),
            f"{imname}_out.jpg"
        )

        cv2.imwrite(output_path, im)
        
