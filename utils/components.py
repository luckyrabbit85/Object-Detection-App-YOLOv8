import yaml
import random
import colorsys


def read_yaml(yaml_file):
    return yaml.safe_load(open(yaml_file, encoding="utf-8").read())


def random_light_color():
    hue = random.random()
    saturation = 0.5 + random.random() / 2.0
    lightness = 0.4 + random.random() / 5.0
    red, green, blue = [
        int(255 * i) for i in colorsys.hls_to_rgb(hue, lightness, saturation)
    ]
    return (red, green, blue)
