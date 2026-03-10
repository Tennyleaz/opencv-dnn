import cv2
from cv2.typing import MatLike, Scalar
import numpy as np
import time
import psutil
import matplotlib.pyplot as plt


def letterbox(img: MatLike, new_shape=(640, 640), color: Scalar = (114, 114, 114)):
    shape = img.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(shape[1] * ratio), int(shape[0] * ratio))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh)), int(round(dh))
    left, right = int(round(dw)), int(round(dw))
    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img_padded, ratio, left, top


def detect_objects(frame: MatLike, display_width=800):
    # Load dnn model
    model_path = "model/yolov8n.onnx"
    net = cv2.dnn.readNet(model_path)

    # Load the name dataset
    with open("model/coco.names.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # Use CPU
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    input_size = 640
    # Measure start memory & time
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024**2)
    start_time = time.time()
    # Preprocess
    img_letterboxed, ratio, dw, dh = letterbox(
        frame, new_shape=(input_size, input_size)
    )
    blob = cv2.dnn.blobFromImage(
        img_letterboxed, 1 / 255.0, (input_size, input_size), swapRB=True, crop=False
    )
    net.setInput(blob)
    # Inference
    outputs = net.forward()
    # Measure end time & memory
    end_time = time.time()
    mem_after = process.memory_info().rss / (1024**2)
    inference_time = end_time - start_time
    mem_used = mem_after - mem_before
    # Post-processing
    H, W, _ = frame.shape
    boxes, confidences, class_ids = [], [], []
    # Handle output shape variations
    if len(outputs.shape) == 3 and outputs.shape[1] == 25200 and outputs.shape[2] == 85:
        output = outputs[0]
    elif (
        len(outputs.shape) == 3 and outputs.shape[1] == 84 and outputs.shape[2] == 8400
    ):
        output = outputs[0].T
    elif (
        len(outputs.shape) == 3 and outputs.shape[1] == 8400 and outputs.shape[2] == 85
    ):
        output = outputs[0]
    else:
        raise ValueError(f"Unexpected output shape: {outputs.shape}")
    for detection in output:
        if output.shape[1] == 85:
            scores = detection[5:]
            objectness = detection[4]
            class_id = np.argmax(scores)
            confidence = objectness * scores[class_id]
        elif output.shape[1] == 84:
            scores = detection[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
        else:
            continue
        if confidence > 0.3:
            cx, cy, w, h = detection[0:4]
            cx *= input_size
            cy *= input_size
            w *= input_size
            h *= input_size
            x = int((cx - w / 2 - dw) / ratio)
            y = int((cy - h / 2 - dh) / ratio)
            w = int(w / ratio)
            h = int(h / ratio)
            x = max(0, x)
            y = max(0, y)
            w = min(W - x, w)
            h = min(H - y, h)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    if len(indices) > 0:
        # Draw Boxes and Labels with white background + larger text
        for i in indices:
            x, y, w, h = boxes[i]
            label = class_names[class_ids[i]]
            confidence = confidences[i]
            color = (0, 255, 0)
            # Text setup
            label_text = f"{label} {confidence:.2f}"
            font_scale = 2.0
            font_thickness = 3
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            # Position text above box
            text_x = x
            text_y = y - 10
            if text_y < text_height:
                text_y = y + h + text_height + 10  # move below if near top
            # Draw white background rectangle
            cv2.rectangle(
                frame,
                (text_x, text_y - text_height - baseline),
                (text_x + text_width, text_y + baseline),
                (255, 255, 255),
                cv2.FILLED,
            )
            # Draw black text on white background
            cv2.putText(
                frame,
                label_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thickness,
            )
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 4)
    # Print performance stats
    print(f" Inference Time: {inference_time:.3f} seconds")
    print(f" Memory Used: {mem_used:.3f} MB")
    # Resize for display (maintain aspect ratio)
    h, w = frame.shape[:2]
    scale = display_width / w
    display_height = int(h * scale)
    resized_frame = cv2.resize(frame, (display_width, display_height))
    return resized_frame


def main():
    image = cv2.imread("test.jpg")
    if image is None:
        print("Cannot load image!")
        return

    result = detect_objects(image)
    # Display image in Jupyter Notebook
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(result_rgb)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
