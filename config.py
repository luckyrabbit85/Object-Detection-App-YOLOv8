from pathlib import Path

ROOT_PATH = Path.cwd()  # current working directory
LOCAL_MODEL_PATH = ROOT_PATH.joinpath("models")  # path to the models directory
LOCAL_FONT_PATH = ROOT_PATH.joinpath("fonts")  # path to the fonts directory
EDO_PATH = ROOT_PATH / "fonts" / "edo.ttf"  # path of the edo.ttf font file
