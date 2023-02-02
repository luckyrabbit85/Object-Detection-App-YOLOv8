"""
Module to install the desired fonts if not already present.

The module provides two functions: install_fonts and download_and_save_fonts.

The install_fonts function is used to install the fonts if not already present. It checks if the font directory exists and then determines if the desired fonts are present. 
If the desired fonts are not present, the download_and_save_fonts function is called to download and save the fonts.

The download_and_save_fonts function is used to download and save the fonts to the specified directory.

Attributes:
    console (rich.console.Console): Initializes the console for output.
    config.DESIRED_FONT (list): List of desired fonts to install.
    configFONT_DOWNLOAD_URLS (dict): Dictionary of URLs of fonts to download.

Functions:
    install_fonts(fonts_directory: Path): This function is used to install the fonts if not already present.
    download_and_save_fonts(fonts_to_install: list, fonts_directory: Path): This function is used to download and save the fonts.

"""

import sys
from pathlib import Path
from glob import glob
import gdown
from typing import List
import config
from config import console


def install_fonts(fonts_directory) -> None:
    """
    This function is used to install the fonts if not already present.

    Args:
        fonts_directory (Path): Path to the font directory.

    Returns:
        None
    """
    # Check if the font directory exists
    fonts_directory = Path(fonts_directory)
    if fonts_directory.is_dir():
        existing_fonts = [font.name for font in fonts_directory.glob("*.ttf")]
        # Get the list of fonts that need to be installed
        fonts_to_install = list(
            set(config.DESIRED_FONT).difference(set(existing_fonts))
        )
        # If there are fonts to install
        if fonts_to_install != []:
            console.print(
                f"{fonts_to_install} font does not exist, downloading...", style="red"
            )
            # Download and save the fonts
            download_and_save_fonts(fonts_to_install, fonts_directory)
        else:
            console.print(f"{config.DESIRED_FONT} font already exists!", style="green")
    else:
        console.print(
            "Font library does not exist and is being created...", style="yellow"
        )
        fonts_directory.mkdir()
        download_and_save_fonts(config.DESIRED_FONT, fonts_directory)


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
        url = config.FONT_DOWNLOAD_URLS[f"{font}"]
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
