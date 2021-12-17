import sys, os
from PIL import Image


def roll(image, delta):
    """Roll an image sideways."""
    xsize, ysize = image.size

    delta = delta % xsize
    if delta == 0:
        return image

    part1 = image.crop((0, 0, delta, ysize))
    part2 = image.crop((delta, 0, xsize, ysize))
    image.paste(part1, (xsize - delta, 0, xsize, ysize))
    image.paste(part2, (0, 0, xsize - delta, ysize))

    return image


for infile in sys.argv[1:]:
    try:
        with Image.open(infile) as im:
            print("kuckuck!")
            r, g, b = im.split()
            im = Image.merge("RGB", (b, g, r))
            im.show()
    except OSError:
        pass


# lenna = Image.open("lenna.png")
