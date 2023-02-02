import yaml
import random
import argparse
import colorsys
import gradio as gr
from pathlib import Path
from copy import deepcopy
from ultralytics import YOLO
from collections import Counter
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from util.font_downloader import install_fonts
from util.model_downloader import search_models


ROOT_PATH = Path.cwd()  # current working directory
LOCAL_MODEL_PATH = ROOT_PATH.joinpath("models")  # path to the models directory
LOCAL_FONT_PATH = ROOT_PATH.joinpath("fonts")  # path to the fonts directory

print(LOCAL_MODEL_PATH)
search_models(LOCAL_MODEL_PATH)  # Download models for detection
install_fonts(LOCAL_FONT_PATH)  # Install fonts for GUI display
EDO_PATH = ROOT_PATH / "fonts" / "edo.ttf"  # path of the edo.ttf font file
# FontProperties object for the edo font file
edo = font_manager.FontProperties(fname=EDO_PATH)
# TrueType font object for the edo font file with a size of 20
textFont = ImageFont.truetype(str(EDO_PATH), size=20)
# list of strings containing the different object style
obj_style = ["Small target", "Medium target", "Big target"]


def parse_args(known=False):
    """
    This function sets up command line arguments to be passed to the script.
    It uses the argparse module to define and parse the arguments.
    Parameters:
        known (bool): if True, only the known arguments will be parsed.
    Returns:
        args (Namespace): the parsed arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser(description="Gradio YOLOv8 Det v0.1")
    parser.add_argument(
        "--source", "-src", default="upload", type=str, help="image input source"
    )
    parser.add_argument(
        "--source_video",
        "-src_v",
        default="upload",
        type=str,
        help="video input source",
    )
    parser.add_argument(
        "--img_tool", "-it", default="editor", type=str, help="input image tool"
    )
    parser.add_argument(
        "--model_name", "-mn", default="yolov8s", type=str, help="model name"
    )
    parser.add_argument(
        "--model_cfg", "-mc", default="model_names.yaml", type=str, help="model config"
    )
    parser.add_argument(
        "--class_names", "-cls", default="class_names.yaml", type=str, help="cls name"
    )
    parser.add_argument(
        "--nms_conf",
        "-conf",
        default=0.5,
        type=float,
        help="model NMS confidence threshold",
    )
    parser.add_argument(
        "--nms_iou", "-iou", default=0.45, type=float, help="model NMS IoU threshold"
    )
    parser.add_argument(
        "--device", "-dev", default="cuda:0", type=str, help="cuda or cpu"
    )
    parser.add_argument(
        "--inference_size", "-isz", default=640, type=int, help="model inference size"
    )
    parser.add_argument(
        "--max_detnum", "-mdn", default=50, type=float, help="model max det num"
    )
    parser.add_argument(
        "--slider_step", "-ss", default=0.05, type=float, help="slider step"
    )

    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


def read_yaml(yaml_file):
    """
    This function reads and parses a YAML file.
    Parameters:
        yaml_file (str): the path of the YAML file to read.
    Returns:
        data (Any): the contents of the YAML file, parsed into a Python data structure.
    """
    return yaml.safe_load(open(yaml_file, encoding="utf-8").read())


def random_light_color():
    """
    This function generates a random light color in RGB format.
    it uses the random and colorsys module to generate random color.
    Returns:
        A tuple of integers representing the color in the RGB format.
    """
    hue = random.random()
    saturation = 0.5 + random.random() / 2.0
    lightness = 0.4 + random.random() / 5.0
    red, green, blue = [
        int(255 * i) for i in colorsys.hls_to_rgb(hue, lightness, saturation)
    ]
    return (red, green, blue)


def draw_detections(
    image_path, scores, bounding_boxes, labels, class_indices, text_font, class_colors
):
    """
    The function `draw_detections` is used to draw bounding boxes, labels, and scores on an image, given the input parameters.
    It takes in the following parameters:
        - image_path: a string representing the path to the image file
        - scores: a list of scores (floats) associated with each detection
        - bounding_boxes: a list of bounding boxes (tuples of 4 integers representing the x, y coordinates
          of the top left corner and bottom right corner of the box) associated with each detection
        - labels: a list of labels (strings) associated with each detection
        - class_indices: a list of class indices (integers) associated with each detection
        - text_font: a font object used for drawing text on the image
        - class_colors: a dictionary that maps class indices to colors
    The function returns an image object with the detections drawn on it.
    """
    image = Image.open(image_path)  # Open image file
    draw = ImageDraw.Draw(image)  # Create an ImageDraw object
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


def detect_image_with_yolo(
    img_path, model_name, infer_size, confidence_threshold, iou_threshold
):
    # Declaring global variable
    global class_names_list
    # Load the pre-trained YOLO model
    model = YOLO(f"{LOCAL_MODEL_PATH}/{model_name}.pt")
    # Make predictions on the input image
    predictions = model.predict(
        source=img_path,
        imgsz=infer_size,
        conf=confidence_threshold,
        iou=iou_threshold,
        return_outputs=True,
    )
    # Convert the predictions to a list
    detection_list = []
    for i in predictions:
        detection_list.append(deepcopy(i))
    detections = detection_list[0]["det"].tolist()
    # Initialize variables to keep track of object sizes and counts
    small_obj, medium_obj, large_obj = 0, 0, 0
    obj_areas = []
    confidences = []
    bounding_boxes = []
    classes = []
    class_indices = []
    # Check if any objects were detected
    if detections != []:
        # Extract information from the predictions
        for i in range(len(detections)):
            class_index = int(detections[i][5])
            class_name = class_names_list[class_index]
            class_indices.append(class_index)
            classes.append(class_name)

            x0 = int(detections[i][0])
            y0 = int(detections[i][1])
            x1 = int(detections[i][2])
            y1 = int(detections[i][3])

            bounding_boxes.append((x0, y0, x1, y1))

            conf = float(detections[i][4])
            confidences.append(conf)

            width = x1 - x0
            height = y1 - y0
            area = width * height
            obj_areas.append(area)

        color_map = {}
        for class_index in list(set(class_indices)):
            color_map[class_index] = random_light_color()
        # Draw detections on the image and save it
        image_with_detections = draw_detections(
            img_path,
            confidences,
            bounding_boxes,
            classes,
            class_indices,
            textFont,
            color_map,
        )
        # Count number of objects in different categories
        for i in range(len(obj_areas)):
            if 0 < obj_areas[i] <= 32**2:
                small_obj += 1
            elif 32**2 < obj_areas[i] <= 96**2:
                medium_obj += 1
            elif obj_areas[i] > 96**2:
                large_obj += 1
        # Compute ratio of object sizes
        total_objects = small_obj + medium_obj + large_obj
        obj_size_ratio = {}
        obj_size_ratio = {
            obj_style[i]: [small_obj, medium_obj, large_obj][i] / total_objects
            for i in range(3)
        }
        # Compute ratio of class labels
        class_ratio = {}
        class_count = Counter(classes)
        class_count_sum = sum(class_count.values())
        for class_name, count in class_count.items():
            class_ratio[class_name] = count / class_count_sum
        # Return the annotated image, object size ratio, and class ratio
        return image_with_detections, obj_size_ratio, class_ratio
    else:
        print("No objects detected in image!")
        return None, None, None


def main(args):
    # Close all open widgets created by the gr library
    gr.close_all()
    # Declare class_names_list as global variable
    global class_names_list
    # Extract the values of the parameters from the args object
    source = args.source
    img_tool = args.img_tool
    nms_conf = args.nms_conf
    nms_iou = args.nms_iou
    model_name = args.model_name
    model_cfg = args.model_cfg
    class_names = args.class_names
    inference_size = args.inference_size
    slider_step = args.slider_step
    # Read the list of available model names from the model_cfg file
    model_names_list = read_yaml(model_cfg)["model_names"]
    # Read the list of class names from the class_names file
    class_names_list = read_yaml(class_names)["class_names"]
    # Create an input widget for the original image
    input_img = gr.Image(
        image_mode="RGB",
        source=source,
        tool=img_tool,
        type="filepath",
        label="Original image",
    )
    # Create a dropdown widget for selecting the model
    input_model = gr.Dropdown(choices=model_names_list, value=model_name, label="Model")
    # Create a slider widget for selecting the inference size
    input_size = gr.Slider(
        384, 1536, step=128, value=inference_size, label="Inference size"
    )
    # Create a slider widget for selecting the confidence threshold
    input_conf = gr.Slider(
        0, 1, step=slider_step, value=nms_conf, label="Confidence threshold"
    )
    # Create a slider widget for selecting the IoU threshold
    input_iou = gr.Slider(0, 1, step=slider_step, value=nms_iou, label="IoU Threshold")
    # Create a list of all input widgets
    inputs_img_list = [input_img, input_model, input_size, input_conf, input_iou]
    # Create an output widget for displaying the results image
    outputs_img = gr.Image(type="pil", label="Image Results")
    # Create an output widget for displaying the object size ratio
    outputs_objSize = gr.Label(label="Target size proportion statistics")
    # Create an output widget for displaying the class ratio
    outputs_clsSize = gr.Label(label="Category detection proportion statistics")
    # Create a list of all output widgets
    outputs_img_list = [outputs_img, outputs_objSize, outputs_clsSize]
    # title
    title = "Gradio YOLOv8 Det"
    # describe
    description = "Author: Aman Roland  \nThis object detection app uses the YOLO (You Only Look Once) algorithm to detect objects within an image or video stream. The app is built using the Gradio and Ultralytics libraries. Thank you to the teams at [Gradio](https://github.com/gradio-app/gradio) & [YOLOv8](https://github.com/ultralytics/ultralytics) for their contributions to this project."
    # sample image
    examples_imgs = [
        [
            "./examples_imgs/beatles.jpg",
            "yolov8s",
            1024,
            0.6,
            0.5,
        ],
        [
            "./examples_imgs/surfing.jpg",
            "yolov8l",
            320,
            0.5,
            0.45,
        ],
        [
            "./examples_imgs/elephant.jpg",
            "yolov8m",
            640,
            0.6,
            0.6,
        ],
        [
            "./examples_imgs/working.jpg",
            "yolov8x",
            1280,
            0.5,
            0.5,
        ],
    ]
    gr_app = gr.Interface(
        fn=detect_image_with_yolo,
        inputs=inputs_img_list,
        outputs=outputs_img_list,
        title=title,
        description=description,
        examples=examples_imgs,
        cache_examples=False,
        flagging_dir="run",
        allow_flagging="manual",
        flagging_options=["good", "generally", "bad"],
        css=".gradio-container {background-image: url('file=./background/background.jpg')",
    )
    gr_app.launch(
        inbrowser=True,
        show_tips=True,
        share=False,
        favicon_path="./favicon/favicon.ico",  # web icon
        show_error=True,  # show error message in browser console
        quiet=True,  # suppress most print statements
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
