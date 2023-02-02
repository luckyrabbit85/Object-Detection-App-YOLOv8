import config
from copy import deepcopy
from ultralytics import YOLO
from collections import Counter
from matplotlib import font_manager
from utils.components import random_light_color
from PIL import Image, ImageDraw, ImageFont


def draw_detections(
    image_path, scores, bounding_boxes, labels, class_indices, class_colors
):
    image = Image.open(image_path)  # Open image file
    draw = ImageDraw.Draw(image)  # Create an ImageDraw object
    font_manager.FontProperties(fname=config.EDO_PATH)
    text_font = ImageFont.truetype(str(config.EDO_PATH), size=20)
    # Iterate through detections
    for index, (score, bbox, label, class_index) in enumerate(
        zip(scores, bounding_boxes, labels, class_indices)
    ):
        # Draw rectangle for the bounding box
        draw.rectangle(bbox, fill=None, outline=class_colors[class_index], width=2)
        # Create detection message
        detection_message = f"{index+1}-{label} {round(score, 2)}"
        # Get the bounding box for the detection message
        text_box = text_font.getbbox(detection_message)
        text_width, text_height = text_box[2] - text_box[0], text_box[3] - text_box[1]
        label_background = (
            bbox[0],
            bbox[1],
            bbox[0] + text_width,
            bbox[1] + text_height,
        )
        # Draw rectangle for the background of the detection message
        draw.rectangle(
            label_background,
            fill=class_colors[class_index],
            outline=class_colors[class_index],
        )
        # Draw the detection message on the image
        draw.multiline_text(
            (bbox[0], bbox[1]),
            detection_message,
            fill=(0, 0, 0),
            font=text_font,
            align="center",
        )
    # Return the image with the detections drawn on it
    return image


def detect_objects(
    img_path,
    model_name,
    infer_size,
    confidence_threshold,
    iou_threshold,
    class_names_list,
):
    # Load the pre-trained YOLO model
    model = YOLO(f"{config.LOCAL_MODEL_PATH}/{model_name}.pt")
    # Make predictions on the input image
    predictions = model.predict(
        source=img_path,
        imgsz=infer_size,
        conf=confidence_threshold,
        iou=iou_threshold,
        return_outputs=True,
    )
    detection_list = []
    for i in predictions:
        detection_list.append(deepcopy(i))
    detections = detection_list[0]["det"].tolist()
    return extract_info(img_path, detections, class_names_list)


def extract_info(img_path, detections, class_names_list):
    if detections == []:
        print("No objects detected in image!")
        return None, None, None

    obj_areas, confidences, bounding_boxes, classes, class_indices = [], [], [], [], []
    for detection in detections:
        class_index = int(detection[5])
        class_name = class_names_list[class_index]
        class_indices.append(class_index)
        classes.append(class_name)

        x0, y0, x1, y1 = map(int, detection[:4])
        bounding_boxes.append((x0, y0, x1, y1))
        conf = float(detection[4])
        confidences.append(conf)
        obj_areas.append((x1 - x0) * (y1 - y0))

    color_map = {
        class_index: random_light_color() for class_index in set(class_indices)
    }
    small_obj, medium_obj, large_obj = 0, 0, 0
    for area in obj_areas:
        if 0 < area <= 32**2:
            small_obj += 1
        elif 32**2 < area <= 96**2:
            medium_obj += 1
        elif area > 96**2:
            large_obj += 1

    obj_style = ["Small target", "Medium target", "Big target"]
    total_objects = small_obj + medium_obj + large_obj
    obj_size_ratio = {
        style: count / total_objects
        for style, count in zip(obj_style, [small_obj, medium_obj, large_obj])
    }

    class_count = Counter(classes)
    class_ratio = {
        class_name: count / sum(class_count.values())
        for class_name, count in class_count.items()
    }
    image_with_detections = draw_detections(
        img_path,
        confidences,
        bounding_boxes,
        classes,
        class_indices,
        color_map,
    )
    return (
        image_with_detections,
        obj_size_ratio,
        class_ratio,
    )
