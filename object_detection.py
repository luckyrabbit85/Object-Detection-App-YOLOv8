import config
import argparse
import gradio as gr
from utils.components import read_yaml
from utils.detector import detect_objects
from utils.font_downloader import install_fonts
from utils.model_downloader import search_models


def parse_args(known=False):
    """
    parse_args - parses the command line arguments for Gradio YOLOv8 Detection v1.1

    This function creates an ArgumentParser object, adds arguments to it and then returns the parsed arguments.

    Parameters:
    known (bool, optional): If True, returns only known arguments, else returns all arguments. Default is False.

    Returns:
    Namespace: The Namespace object containing the parsed arguments.

    Arguments:
    --source, -src (str, default='upload'): Image input source.
    --source_video, -src_v (str, default='upload'): Video input source.
    --img_tool, -it (str, default='editor'): Input image tool.
    --model_name, -mn (str, default='yolov8s'): Model name.
    --model_cfg, -mc (str, default='model_names.yaml'): Model config file.
    --class_names, -cls (str, default='class_names.yaml'): Class names file.
    --nms_conf, -conf (float, default=0.5): NMS confidence threshold.
    --nms_iou, -iou (float, default=0.45): NMS IoU threshold.
    --device, -dev (str, default='cuda:0'): Device to use (cuda or cpu).
    --inference_size, -isz (int, default=640): Inference size.
    --max_detnum, -mdn (float, default=50): Maximum detection number.
    --slider_step, -ss (float, default=0.05): Slider step.
    """
    parser = argparse.ArgumentParser(description="Gradio YOLOv8 Detection v1.1")
    parser.add_argument(
        "--source", "-src", default="upload", type=str, help="Image input source"
    )
    parser.add_argument(
        "--source_video",
        "-src_v",
        default="upload",
        type=str,
        help="Video input source",
    )
    parser.add_argument(
        "--img_tool", "-it", default="editor", type=str, help="Input image tool"
    )
    parser.add_argument(
        "--model_name", "-mn", default="yolov8s", type=str, help="Model name"
    )
    parser.add_argument(
        "--model_cfg",
        "-mc",
        default="model_names.yaml",
        type=str,
        help="Model config file",
    )
    parser.add_argument(
        "--class_names",
        "-cls",
        default="class_names.yaml",
        type=str,
        help="Class names file",
    )
    parser.add_argument(
        "--nms_conf", "-conf", default=0.5, type=float, help="NMS confidence threshold"
    )
    parser.add_argument(
        "--nms_iou", "-iou", default=0.45, type=float, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--device",
        "-dev",
        default="cuda:0",
        type=str,
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--inference_size", "-isz", default=640, type=int, help="Inference size"
    )
    parser.add_argument(
        "--max_detnum", "-mdn", default=50, type=float, help="Maximum detection number"
    )
    parser.add_argument(
        "--slider_step", "-ss", default=0.05, type=float, help="Slider step"
    )

    return parser.parse_known_args()[0] if known else parser.parse_args()


def gradio_app(args):
    """
    A function to create and launch a Gradio object detection app using YOLOv8.

    The function takes an object `args` as an input, which should contain the following attributes:
        - `source` (str): A string specifying the source of the input image (e.g. 'filepath', 'url', etc.).
        - `img_tool` (str): A string specifying the image tool (e.g. 'PIL', 'OpenCV', etc.).
        - `model_cfg` (str): A string specifying the path to the model configuration file in YAML format.
        - `class_names` (str): A string specifying the path to the class names file in YAML format.
        - `model_name` (str): A string specifying the name of the YOLOv8 model to use for object detection.
        - `inference_size` (int): An integer specifying the size of the input image for inference.
        - `nms_conf` (float): A float specifying the confidence threshold for non-maximum suppression.
        - `nms_iou` (float): A float specifying the IoU threshold for non-maximum suppression.
        - `slider_step` (float): A float specifying the step size for the slider widgets.

    The function creates an instance of the `Gradio` interface, sets the inputs and outputs, sets the title and description, and launches the app in the browser. The app allows the user to upload an image, select the YOLOv8 model, set the inference size, and adjust the confidence and IoU thresholds for non-maximum suppression. The app then displays the resulting object detections.

    Returns:
        None
    """
    gr.close_all()
    model_names_list = read_yaml(args.model_cfg)["model_names"]
    class_names_list = read_yaml(args.class_names)["class_names"]

    # Create a list of all input widgets
    input_widgets_list = [
        gr.Image(
            image_mode="RGB",
            source=args.source,
            tool=args.img_tool,
            type="filepath",
            label="Original image",
        ),
        gr.Dropdown(choices=model_names_list, value=args.model_name, label="Model"),
        gr.Slider(
            384, 1536, step=128, value=args.inference_size, label="Inference size"
        ),
        gr.Slider(
            0,
            1,
            step=args.slider_step,
            value=args.nms_conf,
            label="Confidence threshold",
        ),
        gr.Slider(
            0, 1, step=args.slider_step, value=args.nms_iou, label="IoU Threshold"
        ),
        gr.Dropdown(value=class_names_list, visible=False),
    ]

    # Create a list of all output widgets
    oytput_widgets_list = [
        gr.Image(type="pil", label="Image Results"),
        gr.Label(label="Target size proportion statistics"),
        gr.Label(label="Category detection proportion statistics"),
    ]
    # Title
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
        fn=detect_objects,
        inputs=input_widgets_list,
        outputs=oytput_widgets_list,
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
    search_models(config.LOCAL_MODEL_PATH)  # Download models for detection
    install_fonts(config.LOCAL_FONT_PATH)  # Install fonts for GUI display
    args = parse_args()
    gradio_app(args)
