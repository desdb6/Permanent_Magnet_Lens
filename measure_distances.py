"""
Author: Des De Borger

Script to measure distances on images.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider

WORKING_DIRECTORY = Path(r'01042026_images\np\Resolution_np_shadow')
PIXEL_SIZE = 55e-6

image_path_list = [file for file in WORKING_DIRECTORY.glob('**/*.txt') if 'measured' not in str(file)]
image_index = 0
coords = []

fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.25) # Make room for the slider

# Create the slider axes [left, bottom, width, height]
slider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
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
    global img_display, slider, coords

    ax.clear()
    coords = [] # Clear clicks when switching images
    image_path = image_path_list[index]
    img = np.loadtxt(image_path)
    wd = extract_working_distance(image_path)

    img_display = ax.imshow(img, cmap="grey")
    ax.set_title(f"Working distance: {wd}, index: {index}")

    # Setup/Update Slider
    img_min, img_max = 0, np.max(img)
    slider_ax.clear()
    slider = RangeSlider(slider_ax, 'Threshold', img_min, img_max, valinit=(img_min, img_max))
    
    def update(val):
        img_display.set_clim(val[0], val[1])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.draw()

def on_key(event):
    global image_index
    if event.key == 'right':
        image_index = (image_index + 1) % len(image_path_list)
        load_image(image_index)
    elif event.key == 'left':
        image_index = (image_index - 1) % len(image_path_list)
        load_image(image_index)

def on_click(event):
    if event.inaxes != ax: return
    
    coords.append((event.xdata, event.ydata))
    ax.plot(event.xdata, event.ydata, 'rx')
    
    if len(coords) == 2:
        dist = np.sqrt((coords[1][0]-coords[0][0])**2 + (coords[1][1]-coords[0][1])**2)*PIXEL_SIZE
        
        # Save logic
        current_path = image_path_list[image_index]
        log_name = distance_file_path(current_path)
        with open(log_name, "a") as f:
            f.write(f"Dist: {dist} m\n")
        
        ax.plot([coords[0][0], coords[1][0]], [coords[0][1], coords[1][1]], 'r--')
        coords.clear()
    plt.draw()


if __name__ == "__main__":
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('button_press_event', on_click)
    load_image(0)
    plt.show()
    