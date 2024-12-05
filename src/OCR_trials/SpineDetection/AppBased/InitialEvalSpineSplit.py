# Based on work done here: https://github.com/LakshyaKhatri/Bookshelf-Reader-API

from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import math

def remove_duplicate_lines(sorted_points):
    '''
    Serches for the lines that are drawn
    over each other in the image and returns
    a list of non duplicate line co-ordinates
    '''
    last_x1 = 0
    non_duplicate_points = []
    for point in sorted_points:
        ((x1, y1), (x2, y2)) = point
        if last_x1 == 0:
            non_duplicate_points.append(point)
            last_x1 = x1

        elif abs(last_x1 - x1) >= 25:
            non_duplicate_points.append(point)
            last_x1 = x1

    return non_duplicate_points


def get_points_in_x_and_y(hough_lines, max_y):
    '''
    Takes a list of trigonometric form of lines
    and returns their starting and ending
    co-ordinates
    '''
    points = []
    for line in hough_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + (max_y + 100) * (-b))
        y1 = int(y0 + (max_y + 100) * (a))
        start = (x1, y1)

        x2 = int(x0 - (max_y + 100) * (-b))
        y2 = int(y0 - (max_y + 100) * (a))
        end = (x2, y2)

        points.append((start, end))

    # Add a line at the very end of the image
    points.append(((500, max_y), (500, 0)))

    return points


def shorten_line(points, y_max):
    '''
    Takes a list of starting and ending
    co-ordinates of different lines
    and returns their trimmed form matching
    the image height
    '''
    shortened_points = []
    for point in points:
        ((x1, y1), (x2, y2)) = point

        # Slope
        try:
            m = (y2 - y1) / (x2 - x1)
        except ZeroDivisionError:
            m = -1  # Infinite slope

        if m == -1:
            shortened_points.append(((x1, y_max), (x1, 0)))
            continue

        # From equation of line:
        # y-y1 = m (x-x1)
        # x = (y-y1)/m + x1
        # let y = y_max
        new_x1 = math.ceil(((y_max - y1) / m) + x1)
        start_point = (abs(new_x1), y_max)

        # Now let y = 0
        new_x2 = math.ceil(((0 - y1) / m) + x1)
        end_point = (abs(new_x2), 0)

        shortened_points.append((start_point, end_point))

    return shortened_points


def get_cropped_images(image, points):
    '''
    Takes a spine line drawn image and
    returns a list of opencv images splitted
    from the drawn lines
    '''
    image = image.copy()
    y_max, _, _ = image.shape
    last_x1 = 0
    last_x2 = 0
    cropped_images = []

    for point in points:
        ((x1, y1), (x2, y2)) = point

        crop_points = np.array([[last_x1, y_max],
                                [last_x2, 0],
                                [x2, y2],
                                [x1, y1]])

        # Crop the bounding rect
        rect = cv2.boundingRect(crop_points)
        x, y, w, h = rect
        cropped = image[y: y + h, x: x + w].copy()

        # make mask
        crop_points = crop_points - crop_points.min(axis=0)
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [crop_points], -1, (255, 255, 255), -1, cv2.LINE_AA)

        # do bit-op
        dst = cv2.bitwise_and(cropped, cropped, mask=mask)
        cropped_images.append(dst)

        last_x1 = x1
        last_x2 = x2

    return cropped_images


def resize_img(img):
    img = img.copy()
    img_ht, img_wd, _ = img.shape
    ratio = img_wd / img_ht
    new_width = 500
    new_height = math.ceil(new_width / ratio)
    resized_image = cv2.resize(img, (new_width, new_height))

    return resized_image


def detect_spines(img):
    '''
    Returns a list of lines seperating
    the detected spines in the image
    '''
    img = img.copy()
    height, width, _ = img.shape

    blur = cv2.GaussianBlur(img, (5, 5), 0)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    edge = cv2.Canny(gray, 50, 70)

    kernel = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=np.uint8)

    img_erosion = cv2.erode(edge, kernel, iterations=1)
    cv2.imshow("anything", img_erosion)
    cv2.waitKey(0)

    lines = cv2.HoughLines(img_erosion, 1, np.pi / 180, 100)
    if lines is None:
        return []
    points = get_points_in_x_and_y(lines, height)
    points.sort(key=lambda val: val[0][0])
    non_duplicate_points = remove_duplicate_lines(points)

    final_points = shorten_line(non_duplicate_points, height)

    return final_points


def get_spines(img):

    final_image = resize_img(img)
    final_points = detect_spines(final_image)
    cropped_images = get_cropped_images(final_image, final_points)
    return cropped_images


def draw_spine_lines(img):

    final_image = resize_img(img)
    final_points = detect_spines(final_image)
    print(final_points)

    for point in final_points:
        ((x1, y1), (x2, y2)) = point
        final_image = cv2.line(final_image, (x1, y1), (x2, y2), (0, 0, 255), 10)

    return final_image

image_path = "5BooksSameDirections.jpg"
image = cv2.imread(image_path)
img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

output_image = draw_spine_lines(img)
cropped_images = get_spines(img)

i = 1
for img in cropped_images:
    height, width, channels = img.shape
    if width/height < 0.1:
        continue
    cv2.imwrite(f"crops/output_image{i}.jpg", img)
    print(f"crop saved")
    i+=1

cv2.imwrite("output_image.jpg", output_image)
print(f"Output saved")