"""
Module to install the desired fonts if not already present.

The module provides two functions: install_fonts and download_and_save_fonts.

The install_fonts function is used to install the fonts if not already present. It checks if the font directory exists and then determines if the desired fonts are present. 
If the desired fonts are not present, the download_and_save_fonts function is called to download and save the fonts.

The download_and_save_fonts function is used to download and save the fonts to the specified directory.

Attributes:
    console (rich.console.Console): Initializes the console for output.
    desired_fonts (list): List of desired fonts to install.
    urls_of_fonts_to_download (dict): Dictionary of URLs of fonts to download.

Functions:
    install_fonts(fonts_directory: Path): This function is used to install the fonts if not already present.
    download_and_save_fonts(fonts_to_install: list, fonts_directory: Path): This function is used to download and save the fonts.

"""

import sys
from pathlib import Path
from glob import glob
import gdown
from rich.console import Console
from typing import List


# List of desired fonts to install
desired_fonts = ["edo.ttf"]

# List of desired fonts to install
urls_of_fonts_to_download = {
    "edo.ttf": "https://drive.google.com/uc?export=download&id=1hlBHVW5_Aay3tZ9Oqd7NGJ9wcoCZzNgN"
}

# Initialize the console for output
console = Console()


def install_fonts(fonts_directory) -> None:
    """
    This function is used to install the fonts if not already present.

    Args:
        fonts_directory (Path): Path to the font directory.

    Returns:
        None
    """
    # Check if the font directory exists
    font_directory = Path(fonts_directory)
    if font_directory.is_dir():
        existing_fonts = [font.name for font in font_directory.glob("*.ttf")]
        # Get the list of fonts that need to be installed
        fonts_to_install = list(set(desired_fonts).difference(set(existing_fonts)))
        # If there are fonts to install
        if fonts_to_install != []:
            console.print(
                f"{fonts_to_install} font does not exist, downloading...", style="red"
            )
            # Download and save the fonts
            download_and_save_fonts(fonts_to_install, fonts_directory)
        else:
            console.print(f"{desired_fonts} font already exists!", style="green")
    else:
        console.print(
            "Font library does not exist and is being created...", style="yellow"
        )
        font_directory.mkdir()
        download_and_save_fonts(desired_fonts, font_directory)


def download_and_save_fonts(fonts_to_install: List[str], fonts_directory) -> None:
    """
    This function is used to download and save the fonts.

    Args:
        fonts_to_install (list): List of fonts to install.
        fonts_directory (Path): Path to the font directory.

    Returns:
        None
    """
    for font in fonts_to_install:
        url = urls_of_fonts_to_download[f"{font}"]
        file_path = fonts_directory / font
        try:
            gdown.download(url, str(file_path))
            console.print(
                f"{font} font file download is complete! has been saved to: {file_path}",
                style="green",
            )
        except Exception as e:
            console.print("Path error! End of program!", style="red")
            print(e)
            sys.exit()
