#!/usr/bin/env python3

"""Simple example to load and display a test image using opencv."""

import argparse
import argcomplete
import re
import cv2
import glob
import shutil
import numpy as np


def clip(event, x, y, flags, params):
    """
    Mouse callback function.

    Clipping allows the user to remove the overlay to the right of the pointer when the LMB
    is depressed. The overlay would be the colorized prediction tensor which lays over the
    source image. Clipping reveals the background image.
    Args:
        event (int) OpenCV mouse event.
        x (int) x position of the pointer.
        y (int) y position of the pointer.
        flags (int) flags passed by OpenCV.
        params (dict) passed to the callback (Composite object and mouse_down state variable)
    """
    update = False
    image = None

    def clip_image(x):
        _image = params["image0"].copy()
        if x < 0:
            x = 0
        if x >= _image.shape[1]:
            x = _image.shape[1] - 1
        _image[:, x:, :] = params["image1"][:, x:, :]
        _image[:, x, 0] = 255
        return _image

    if event == cv2.EVENT_LBUTTONDOWN:
        image = clip_image(x)
        params["mouse_down"] = True
        update = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if params["mouse_down"]:
            image = clip_image(x)
            update = True

    elif event == cv2.EVENT_LBUTTONUP:
        image = params["image0"].copy()
        params["mouse_down"] = False
        update = True

    if update:
        cv2.imshow("image diff", image)


def show(file0, file1):
    """Display an image using OpenCV."""

    image0 = cv2.imread(file0)
    image1 = cv2.imread(file1)

    if image0 is None:
        print(f"Unable to load {file0}")
        return

    if image1 is None:
        print(f"Unable to load {file1}")
        return

    image0, image1 = reshape(image0, image1)

    show_diff = False
    diff_image = cv2.absdiff(image0, image1)
    max_diff = np.max(diff_image)
    if max_diff == 0:
        print("no difference")
        return
    diff_image = diff_image / max_diff * 255

    nonzero = np.count_nonzero(diff_image)
    print(f"number of non-zero pixels: {nonzero}")
    if nonzero < 45:
        print("no significant difference")
        return

    print(f"max diff: {max_diff}")

    cv2.imshow("image diff", image0)

    mouse_down = False
    cv2.setMouseCallback("image diff", clip,
                         {"image0": image0, "image1": image1, "mouse_down": mouse_down})

    # Poll for events while the window exists.
    while cv2.getWindowProperty("image diff", 0) >= 0:
        key_code = cv2.waitKey(50)

        # ESC or n breaks
        if key_code == 27 or key_code == 110:
            break

        # q quits
        if key_code == 113:  # q
            exit(0)

        if key_code == 117:  # u
            print("u")
            print(f"Copy {file0} {file1}")
            shutil.copyfile(file0, file1)
            break

        # d toggles diff
        if key_code == 100:  # d
            show_diff = not show_diff
            cv2.imshow("image diff", diff_image if show_diff else image0)
            continue

        if key_code != -1:
            print(key_code)

        # Closing the window breaks
        if cv2.getWindowProperty("image diff", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


def reshape(image0, image1):
    shape = (
        max(image0.shape[0], image0.shape[0]),
        max(image0.shape[1], image0.shape[1]),
        max(image0.shape[2], image0.shape[2])
    )
    b0 = np.zeros(shape, dtype=image0.dtype)
    b1 = np.zeros(shape, dtype=image1.dtype)
    b0[:image0.shape[0], :image0.shape[1], :image0.shape[2]] = image0
    b1[:image0.shape[0], :image0.shape[1], :image0.shape[2]] = image1
    image0 = b0
    image1 = b1
    return image0, image1


def get_args():
    parser = argparse.ArgumentParser(description='Generate repro instructions for a given sha')

    parser.add_argument('files', type=str, nargs="*", help='image files to compare (2 required)')
    parser.add_argument('-l', '--list', action='store_true', help='list baseline image pairs')
    parser.add_argument('-b', '--baseline', action='store_true', help='compare testoutput to '
                                                                      'baseline')
    argcomplete.autocomplete(parser)
    parsed_args = parser.parse_args()
    return parsed_args


def get_logs():
    logs = glob.glob("_testoutput/**/exttest*-*-*-*-*.log", recursive=True)
    return logs


def images_from_log(log):
    with open(log, "r") as f:
        lines = f.readlines()
        for line in lines:
            m = re.match(r".*Capturing (\S*) and comparing with (\S*)", line)
            if m:
                yield m.group(1), m.group(2)


def main():
    args = get_args()

    if args.list:
        logs = get_logs()
        pairs = dict()
        for log in logs:
            for image0, image1 in images_from_log(log):
                pairs[image0] = image1

        for image0, image1 in pairs.items():
            print(f"\t{image0} {image1}")
        return

    if args.baseline:
        logs = get_logs()
        pairs = dict()
        for log in logs:
            for image0, image1 in images_from_log(log):
                pairs[image0] = image1

        for image0, image1 in pairs.items():
            print(f"\t{image0} {image1}")
            show(image0, image1)

    elif len(args.files) == 2:
        file0 = args.files[0]
        file1 = args.files[1]

        show(file0, file1)


if __name__ == "__main__":
    main()
