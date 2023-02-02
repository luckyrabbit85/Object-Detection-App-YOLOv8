from pathlib import Path
from rich.console import Console

console = Console()

ROOT_PATH = Path.cwd()  # current working directory

# Font configs and paths
LOCAL_FONT_PATH = ROOT_PATH.joinpath("fonts")  # path to the fonts directory
FONT = "edo.ttf"
DESIRED_FONT = [FONT]  # List of desired Fonts
FONT_DOWNLOAD_URLS = {
    FONT: "https://drive.google.com/uc?export=download&id=1hlBHVW5_Aay3tZ9Oqd7NGJ9wcoCZzNgN"
}
EDO_PATH = ROOT_PATH / "fonts" / "edo.ttf"  # path of the edo.ttf font file

# Model Configs and paths
LOCAL_MODEL_PATH = ROOT_PATH.joinpath("models")  # path to the models directory
DESIRED_MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
MODEL_DOWNLOAD_URLS = {
    "yolov8n": "https://drive.google.com/uc?export=download&id=1RCv4Dolcku1Q7peLk8m-wUR8_vrmzFVa",
    "yolov8s": "https://drive.google.com/uc?export=download&id=1UrN1Uj4dMLBYIz-4ZhyUyWg1TACOlj_7",
    "yolov8m": "https://drive.google.com/uc?export=download&id=13FfHplOIEl1kiVGYpfyJwdVikA-KMK3o",
    "yolov8l": "https://drive.google.com/uc?export=download&id=1vPv7qUNiBDtrdRKMqIu4t413yBbfn7CP",
    "yolov8x": "https://drive.google.com/uc?export=download&id=1CiF2sme6Wo0TW-NQ9BYaBce_uWgipBMs",
}
