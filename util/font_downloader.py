import sys
from pathlib import Path
from glob import glob
import gdown
from rich.console import Console


# Initialize the console for output
console = Console()

# List of desired fonts to install
desired_fonts = ["edo"]

# List of desired fonts to install
urls_of_fonts_to_download = {
    "edo": "https://drive.google.com/uc?export=download&id=1hlBHVW5_Aay3tZ9Oqd7NGJ9wcoCZzNgN"
}

# This function is used to install the fonts if not already present
def install_fonts(fonts_directory):
    # Check if the fonts directory exists
    fonts_directory = Path(fonts_directory)
    if fonts_directory.is_dir():
        existing_fonts = [font.stem for font in fonts_directory.glob("*.ttf")]
        # Get the list of fonts that need to be installed
        fonts_to_install = list(set(desired_fonts).difference(set(existing_fonts)))
        # If there are fonts to install
        if fonts_to_install != []:
            console.print(
                f"[bold red]{fonts_to_install} font does not exist, downloading...[/bold red]"
            )
            # Download and save the fonts
            download_and_save_fonts(fonts_to_install, fonts_directory)
        else:
            console.print(
                f"{desired_fonts} [bold green] font already exists![/bold green]"
            )
    else:
        console.print(
            "[bold red]Font library does not exist and is being created...[/bold red]"
        )
        fonts_directory.mkdir()
        download_and_save_fonts(desired_fonts, fonts_directory)


# This function is used to download and save the fonts
def download_and_save_fonts(fonts_to_install, fonts_directory):
    for font in fonts_to_install:
        url = urls_of_fonts_to_download[f"{font}"]
        file = fonts_directory / font
        try:
            gdown.download(url, str(file))
            console.print(
                f"[bold green] {font} font file download is complete![/bold green] has been saved to: {file}"
            )
        except Exception as e:
            print("Path error! End of program!")
            print(e)
            sys.exit()
