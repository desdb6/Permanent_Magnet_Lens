"""
Author: Des De Borger

Script to open h5 images and export as numpy arrays.
"""

import os
import shutil

import pathlib
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider

def extract_working_distance(path):
    file_name = str(path.stem)
    return file_name.split(' ')[0]

def load_image_numpy(path):
    """Load h5 file into np array."""
    f = h5py.File(path)
    return np.array(f['Frame_0']['Data']).reshape(256, 256)

def convert_np(input_folder, output_folder):
    "Convert all files in folder to numpy in a new folder"
    os.makedirs(output_folder, exist_ok=True)
    p = pathlib.Path(input_folder)
    for image_h5_path in p.glob('**/*.h5'):
        wd = extract_working_distance(image_h5_path)
        image_numpy_array = load_image_numpy(image_h5_path)
        output_path = output_folder + f"/{wd}.txt"
        np.savetxt(output_path, image_numpy_array)
    print(f"Converted {input_folder} to numpy at {output_folder}")

def convert_all_np():
    convert_np(r'01042026_images\h5\Overview mode WD series', r'01042026_images\np\Overview_np_shadow')
    convert_np(r'01042026_images\h5\RESOLUTION mode WD series', r'01042026_images\np\Resolution_np_shadow')
    convert_np(r'01042026_images\h5\Sample above the lens\Overview mode', r'01042026_images\np\Overview_np_image')
    convert_np(r'01042026_images\h5\Sample above the lens\Resolution mode', r'01042026_images\np\Resolution_np_image')

def plot_image(path):
    "Plot h5 image."
    data = np.loadtxt(path)
    plt.imshow(data, cmap='grey')
    plt.show()

def plot_image_scalable(path):
    """Plot h5 image with slider for colorbar limits."""
    data = np.loadtxt(path)
    vmin, vmax = data.min(), data.max()

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)  # make room for slider

    im = ax.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)

    # RangeSlider to control colorbar limits
    slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.04])  # [left, bottom, width, height]
    slider = RangeSlider(slider_ax, 'Limits', vmin, vmax, valinit=(vmin, vmax))

    def update(val):
        im.set_clim(slider.val)
        cbar.update_normal(im)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

def delete_folder(folder_path):
    # Ensure it's a Path object
    folder_path = pathlib.Path(folder_path)

    # Failsafe
    if folder_path in (pathlib.Path("/"), pathlib.Path.home(), pathlib.Path.cwd()):
        raise ValueError("Refusing to delete critical system directory!")

    if not folder_path.exists():
        print(f"Folder {folder_path} does not exist, skipping.")
        return

    # Delete everything inside and the folder itself
    shutil.rmtree(folder_path)
    print(f"Deleted {folder_path}")

def remove_folders():
    delete_folder(pathlib.Path(r'01042026_images\np\Overview_np_shadow'))
    delete_folder(pathlib.Path(r'01042026_images\np\Resolution_np_shadow'))
    delete_folder(pathlib.Path(r'01042026_images\np\Overview_np_image'))
    delete_folder(pathlib.Path(r'01042026_images\np\Resolution_np_image'))

if __name__ == "__main__":
    convert_all_np()
