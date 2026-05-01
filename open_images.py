"""
Author: Des De Borger
Script to measure distances on images.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, Button

WORKING_DIRECTORY = Path('01042026_images/np/Resolution_np_image')
PIXEL_SIZE = 55e-6

image_path_list = [file for file in WORKING_DIRECTORY.glob('**/*.txt') if 'measured' not in str(file)]
image_index = 0
coords = []

fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.25, right=1.05)  # Make room for slider and colorbar

# Colorbar axes (fixed, outside the image)
cbar_ax = fig.add_axes([0.87, 0.25, 0.03, 0.65])
cbar = None

# Slider and button axes
slider_ax = plt.axes([0.20, 0.10, 0.60, 0.03])
button_ax = plt.axes([0.40, 0.03, 0.20, 0.04])
button = Button(button_ax, 'Save PNG')

slider = None
img_display = None


def extract_working_distance(path):
    file_name = str(path.stem)
    return file_name.split(' ')[0]


def distance_file_path(path):
    wd = extract_working_distance(path)
    file_name = str(WORKING_DIRECTORY) + f"/{wd}_mesh_measured_dist.txt"
    return file_name


def load_image(index):
    global img_display, slider, coords, cbar

    ax.clear()
    cbar_ax.clear()
    coords = []

    image_path = image_path_list[index]
    img = np.loadtxt(image_path)
    wd = extract_working_distance(image_path)

    img_display = ax.imshow(img, cmap='gray')
    ax.set_title('Detectorbeeld', fontsize=16)
    ax.set_xlabel('$x$ (px)', fontsize=14)
    ax.set_ylabel(14)

    # Colorbar
    cbar = fig.colorbar(img_display, cax=cbar_ax)
    cbar.set_label('Counts', fontsize=12)

    # Slider
    img_min, img_max = 0, np.max(img)
    slider_ax.clear()
    slider = RangeSlider(slider_ax, 'Limits', img_min, img_max, valinit=(img_min, img_max))

    def update(val):
        img_display.set_clim(val[0], val[1])
        cbar.update_normal(img_display)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.draw()


def save(event):
    current_path = image_path_list[image_index]
    wd = extract_working_distance(current_path)
    vmin, vmax = slider.val
    filename = f'{wd}_vmin{int(vmin)}_vmax{int(vmax)}.png'
    # Hide slider and button for clean export
    slider_ax.set_visible(False)
    button_ax.set_visible(False)
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    slider_ax.set_visible(True)
    button_ax.set_visible(True)
    fig.canvas.draw_idle()
    print(f'Saved: {filename}')


def on_key(event):
    global image_index
    if event.key == 'right':
        image_index = (image_index + 1) % len(image_path_list)
        load_image(image_index)
    elif event.key == 'left':
        image_index = (image_index - 1) % len(image_path_list)
        load_image(image_index)



if __name__ == '__main__':
    button.on_clicked(save)
    fig.canvas.mpl_connect('key_press_event', on_key)
    load_image(0)
    plt.show()