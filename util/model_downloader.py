import sys
from pathlib import Path
from glob import glob
import gdown
from rich.console import Console


# Initialize the console for output
console = Console()

# List of desired models
desired_models = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]

# List of desired models and download urls
urls_of_models_to_download = {
    "yolov8n": "https://drive.google.com/uc?export=download&id=1RCv4Dolcku1Q7peLk8m-wUR8_vrmzFVa",
    "yolov8s": "https://drive.google.com/uc?export=download&id=1UrN1Uj4dMLBYIz-4ZhyUyWg1TACOlj_7",
    "yolov8m": "https://drive.google.com/uc?export=download&id=13FfHplOIEl1kiVGYpfyJwdVikA-KMK3o",
    "yolov8l": "https://drive.google.com/uc?export=download&id=1vPv7qUNiBDtrdRKMqIu4t413yBbfn7CP",
    "yolov8x": "https://drive.google.com/uc?export=download&id=1CiF2sme6Wo0TW-NQ9BYaBce_uWgipBMs",
}

# This function is used to download the models if not already present
def search_models(models_directory):
    # Check if the models directory exists
    models_directory = Path(models_directory)
    if models_directory.is_dir():
        existing_models = [model.stem for model in models_directory.glob("*.pt")]
        print(existing_models)
        # Get the list of models that need to be downloaded
        models_to_download = list(set(desired_models).difference(set(existing_models)))
        print(models_to_download)
        # If there are models to download
        if models_to_download != []:
            console.print(
                f"[bold red]{models_to_download} models does not exist, downloading...[/bold red]"
            )
            # Download and save the models
            download_and_save_models(models_to_download, models_directory)
        else:
            console.print(
                f"{desired_models} [bold green] font already exists![/bold green]"
            )
    else:
        console.print(
            "[bold red]Model library does not exist and is being created...[/bold red]"
        )
        models_directory.mkdir()
        download_and_save_models(desired_models, models_directory)


# This function is used to download and save the models
def download_and_save_models(models_to_download, models_directory):
    for model in models_to_download:
        url = urls_of_models_to_download[f"{model}"]
        file = models_directory.joinpath(f"{model}.pt")
        try:
            gdown.download(url, str(file))
            console.print(
                f"[bold green] {model} font file download is complete![/bold green] has been saved to: {file}"
            )
        except Exception as e:
            print("Path error! End of program!")
            print(e)
            sys.exit()
