import os
import time

from PIL import Image


def concat_img(images, grid_size=2, img_H=512, img_W=512) -> Image:
    grid_img = Image.new("RGB", (grid_size * img_W, grid_size * img_H))
    for row in range(grid_size):
        for col in range(grid_size):
            grid_img.paste(images[grid_size * row + col], (0 + img_W * col, 0 + img_H * row))
    return grid_img


def clear_port(port=6006) -> int:
    export = os.popen("lsof -i :{} | grep {}".format(port, port)).read()
    if export != "":
        export = [word for word in export.split(" ") if word != '']
        os.system("kill -9 {}".format(export[1]))
        print("Successful kill port {}".format(port))
        time.sleep(2)
        return int(export[1])
