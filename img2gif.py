import os
import numpy as np
import glob
import contextlib
from PIL import Image

def img2gif(fp_in, fp_out, sort_key=None):
    with contextlib.ExitStack() as stack:
        # lazily load images
        if sort_key is not None:
            imgs = (stack.enter_context(Image.open(f))
                    for f in sorted(glob.glob(fp_in), key=sort_key))
        else:
            imgs = (stack.enter_context(Image.open(f))
                    for f in sorted(glob.glob(fp_in)))

        # extract  first image from iterator
        img = next(imgs)

        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=100, loop=0)

if __name__ == '__main__':
    try:
        os.mkdir('gifs')
    except FileExistsError:
        pass

    def sort_key(x):
        num = int(x.split('_')[-1].split('.')[0])
        return num

    fp_in = f'graphs_gif/*.png'
    fp_out = f'gifs/graph_structure.gif'

    img2gif(fp_in, fp_out, sort_key=sort_key)
