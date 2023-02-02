import sys
from pathlib import Path
from glob import glob
import gdown
import config
from config import console

# This function is used to download the models if not already present
def search_models(models_directory):
    # Check if the models directory exists
    models_directory = Path(models_directory)
    if models_directory.is_dir():
        existing_models = [model.stem for model in models_directory.glob("*.pt")]
        # Get the list of models that need to be downloaded
        models_to_download = list(
            set(config.DESIRED_MODELS).difference(set(existing_models))
        )
        # If there are models to download
        if models_to_download != []:
            console.print(
                f"{models_to_download} models does not exist, downloading..",
                style="bold red",
            )
            # Download and save the models
            download_and_save_models(models_to_download, models_directory)
        else:
            console.print(
                f"{config.DESIRED_MODELS} models already exists!", style="bold green"
            )
    else:
        console.print(
            "Model library does not exist and is being created..", style="bold yellow"
        )
        models_directory.mkdir()
        download_and_save_models(config.DESIRED_MODELS, models_directory)


# This function is used to download and save the models
def download_and_save_models(models_to_download, models_directory):
    for model in models_to_download:
        url = config.MODEL_DOWNLOAD_URLS[f"{model}"]
        file_path = models_directory.joinpath(f"{model}.pt")
        try:
            gdown.download(url, str(file_path))
            console.print(
                f"[bold green] {model} font file download is complete![/bold green] has been saved to: {file_path}"
            )
        except Exception as e:
            print("Path error! End of program!")
            print(e)
            sys.exit()
