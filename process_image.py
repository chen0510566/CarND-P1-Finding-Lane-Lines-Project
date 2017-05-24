import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


#def process_image(img):

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    #
    left_points = []
    right_points = []

    left_miny = np.shape(img)[0]-1;
    left_maxy = np.shape(img)[0]-1;
    right_miny = np.shape(img)[0]-1;
    right_maxy=np.shape(img)[0]-1;
    for line in lines:
        if line[0][0] < np.shape(img)[1] / 2:
            left_points.append([line[0][0], line[0][1]])
            left_points.append([line[0][2], line[0][3]])
            if line[0][1]<left_miny:
                left_miny=line[0][1]
        else:
            right_points.append([line[0][0], line[0][1]])
            right_points.append([line[0][2], line[0][3]])
            if line[0][1]<right_miny:
                right_miny=line[0][1]

    left_line = cv2.fitLine(np.array([left_points]), cv2.DIST_L2, 0, 0.01, 0.01)
    right_line = cv2.fitLine(np.array([right_points]), cv2.DIST_L2, 0, 0.01, 0.01)

    left_minx = int(left_line[0][0]*(left_miny-left_line[3][0])/left_line[1][0]+left_line[2][0])
    left_maxx = int(left_line[0][0]*(left_maxy-left_line[3][0])/left_line[1][0]+left_line[2][0])

    right_minx = int(right_line[0][0] * (right_miny - right_line[3][0]) / right_line[1][0] + right_line[2][0])
    right_maxx = int(right_line[0][0] * (right_maxy - right_line[3][0]) / right_line[1][0] + right_line[2][0])

    cv2.line(img, (left_minx, left_miny), (left_maxx, left_maxy), color, 5)
    cv2.line(img, (right_minx, right_miny), (right_maxx, right_maxy), color, 5)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, [0, 0, 255])
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_single_image(path):
    raw_img = cv2.imread(path)
    gray_img = grayscale(raw_img)
    blur_img = gaussian_blur(gray_img, 5)
    canny_img = canny(blur_img, 50, 150)
    x_dim = np.shape(raw_img)[1]
    y_dim = np.shape(raw_img)[0]
    vertices = np.array([[(10, y_dim - 1), (x_dim - 10, y_dim - 1), (x_dim / 2 + 40, y_dim / 1.65), (x_dim / 2 - 40, y_dim / 1.65)]],dtype=np.int32)
    sub_img = region_of_interest(canny_img, vertices)
    hough_img = hough_lines(sub_img, 1, 3.14 / 180, 20, 10, 20)
    hough_img = gaussian_blur(hough_img, 5)
    weighted_image = weighted_img(raw_img, hough_img)
    return weighted_image

filenames=os.listdir('test_images/')
for name in filenames:
    file_path='test_images/'+name
    img=process_single_image(file_path)
    cv2.imshow(name + 'weighted', img)
    cv2.imwrite(file_path[0:len(file_path)-4]+'_result.jpg', img)
    # blur_img = gaussian_blur(weighted_image, 5)
    # cv2.imshow(name + 'wb', weighted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


