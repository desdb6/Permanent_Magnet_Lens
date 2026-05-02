"""
Author: Des De Borger
Script to measure distances on images.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, Button

WORKING_DIRECTORY = Path('01042026_images/np/Resolution_np_shadow')
PIXEL_SIZE = 55e-6

image_path_list = [file for file in WORKING_DIRECTORY.glob('**/*.txt') if 'measured' not in str(file)]
image_index = 0
coords = []

fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.25, right=1.05)

cbar_ax = fig.add_axes([0.87, 0.25, 0.03, 0.65])
cbar = None

slider_ax = plt.axes([0.20, 0.10, 0.60, 0.03])
button_ax = plt.axes([0.28, 0.03, 0.18, 0.04])
delete_ax = plt.axes([0.54, 0.03, 0.18, 0.04])

button = Button(button_ax, 'Save PNG')
delete_button = Button(delete_ax, 'Delete distances', color='tomato', hovercolor='red')

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
    # ax.set_title('Detectorbeeld', fontsize=14)
    ax.set_title(f'Working distance: {wd}  |  Image {index + 1}/{len(image_path_list)}', fontsize=14)
    ax.set_xlabel('x (px)', fontsize=12)
    ax.set_ylabel('y (px)', fontsize=12)

    cbar = fig.colorbar(img_display, cax=cbar_ax)
    cbar.set_label('Counts', fontsize=12)

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
    slider_ax.set_visible(False)
    button_ax.set_visible(False)
    delete_ax.set_visible(False)
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    slider_ax.set_visible(True)
    button_ax.set_visible(True)
    delete_ax.set_visible(True)
    fig.canvas.draw_idle()
    print(f'Saved: {filename}')


def delete_distances(event):
    # Remove all lines and text annotations (keep only the image)
    for line in ax.lines[:]:
        line.remove()
    for text in ax.texts[:]:
        text.remove()
    coords.clear()
    # Also delete the saved distances file for this image
    current_path = image_path_list[image_index]
    log_name = Path(distance_file_path(current_path))
    if log_name.exists():
        log_name.unlink()
        print(f'Deleted: {log_name}')
    fig.canvas.draw_idle()


def on_key(event):
    global image_index
    if event.key == 'right':
        image_index = (image_index + 1) % len(image_path_list)
        load_image(image_index)
    elif event.key == 'left':
        image_index = (image_index - 1) % len(image_path_list)
        load_image(image_index)


def on_click(event):
    if event.inaxes != ax:
        return
    coords.append((event.xdata, event.ydata))
    ax.plot(event.xdata, event.ydata, 'rx')
    if len(coords) == 2:
        dist = np.sqrt((coords[1][0] - coords[0][0])**2 +
                       (coords[1][1] - coords[0][1])**2) * PIXEL_SIZE
        current_path = image_path_list[image_index]
        log_name = distance_file_path(current_path)
        with open(log_name, 'a') as f:
            f.write(f'{dist * 1000}\n')
        ax.plot([coords[0][0], coords[1][0]], [coords[0][1], coords[1][1]], 'r--')
        mid_x = (coords[0][0] + coords[1][0]) / 2
        mid_y = (coords[0][1] + coords[1][1]) / 2
        ax.text(mid_x, mid_y, f'{dist*1e6:.1f} µm', color='red', fontsize=10,
                ha='center', va='bottom')
        coords.clear()
    plt.draw()


if __name__ == '__main__':
    button.on_clicked(save)
    delete_button.on_clicked(delete_distances)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('button_press_event', on_click)
    load_image(0)
    plt.show()