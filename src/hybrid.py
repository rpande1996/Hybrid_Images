import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from getImages import *

dir = "../input/"
afghan_before = dir + "Afghan_girl_before.jpg"
afghan_after = dir + "Afghan_girl_after.jpg"
bicycle = dir + "bicycle.bmp"
bird = dir + "bird.bmp"
cat = dir + "cat.bmp"
dog = dir + "dog.bmp"
einstein = dir + "einstein.bmp"
fish = dir + "fish.bmp"
makeup_before = dir + "makeup_before.jpg"
makeup_after = dir + "makeup_after.jpg"
marilyn = dir + "marilyn.bmp"
motorcycle = dir + "motorcycle.bmp"
plane = dir + "plane.bmp"
submarine = dir + "submarine.bmp"

pair1 = "pair1/"
pair2 = "pair2/"
pair3 = "pair3/"
pair4 = "pair4/"
pair5 = "pair5/"
pair6 = "pair6/"
pair7 = "pair7/"

def check_pair(list, boolean):
    im1, im2 = list[0], list[1]
    i1, i2 = check_image(im1, im2, boolean)
    pair = [i1, i2]
    return pair

def gray(image):
    gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gr


def create_pair(string1, string2, boolean):
    pair = []
    im1 = Image.open(string1)
    im2 = Image.open(string2)
    img1 = np.array(im1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2 = np.array(im2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    if boolean:
        img1 = gray(img1)
        img2 = gray(img2)
    img1, img2 = check_image(img1, img2, boolean)
    pair.append(img1)
    pair.append(img2)
    return pair


def check_image(img1, img2, boolean):
    if img1.shape != img2.shape:
        h1, w1 = img1.shape[0], img1.shape[1]
        h2, w2 = img2.shape[0], img2.shape[1]
        diff_h = abs(h1 - h2)
        diff_w = abs(w1 - w2)
        heights = [h1, h2]
        index = heights.index(min(heights))
        if index == 0:
            img = img1
            big = img2
        else:
            img = img2
            big = img1
        if boolean:
            col_add = np.zeros(shape=(img.shape[0], diff_w))
            row_add = np.zeros(shape=(diff_h, diff_w + img.shape[1]))
        else:
            col_add = np.zeros(shape=(img.shape[0], diff_w, img.shape[2]))
            row_add = np.zeros(shape=(diff_h, diff_w + img.shape[1], img.shape[2]))
        change = np.hstack((img, col_add))
        change = np.vstack((change, row_add))
        return big, change
    else:
        return img1, img2


def generateGaussian(size, scaleX, scaleY):
    lower_limit = int(-((size - 1) / 2))
    upper_limit = abs(lower_limit) + 1
    ind = np.arange(lower_limit, upper_limit)
    row = np.reshape(ind, (ind.shape[0], 1)) + np.zeros((1, ind.shape[0]))
    col = np.reshape(ind, (1, ind.shape[0])) + np.zeros((ind.shape[0], 1))
    G = (1 / (2 * np.pi * (scaleX * scaleY))) * np.exp(
        -(((col) ** 2 / (2 * (scaleX ** 2))) + ((row) ** 2 / (2 * (scaleY ** 2)))))
    return G


def vis_hybrid_image(hybrid_image):
    scales = 5
    scale_factor = 0.5
    padding = 5
    original_height = hybrid_image.shape[0]
    num_colors = hybrid_image.shape[2]
    output = hybrid_image
    cur_image = hybrid_image

    for i in range(2, scales):
        output = np.concatenate((output, np.ones((original_height, padding, num_colors), dtype=int)), axis=1)
        width = int(cur_image.shape[1] * scale_factor)
        height = int(cur_image.shape[0] * scale_factor)
        dim = (width, height)
        cur_image = cv2.resize(cur_image, dim, interpolation=cv2.INTER_LINEAR)
        tmp = np.concatenate(
            (np.ones((original_height - cur_image.shape[0], cur_image.shape[1], num_colors)), cur_image), axis=0)
        output = np.concatenate((output, tmp), axis=1)

    output = (output * 255).astype(np.uint8)
    return output


def getgraypair(pair):
    pair[0] = gray(pair[0])
    pair[1] = gray(pair[1])
    return pair


def getHybrid(pair):
    highpass_kernel = generateGaussian(5, 2, 2)
    lowpass_kernel = generateGaussian(7, 3, 3)
    blurred_high = cv2.filter2D(pair[0], -1, highpass_kernel)
    blurred_low = cv2.filter2D(pair[1], -1, lowpass_kernel)
    blurred_low = blurred_low.astype(np.uint8)
    high = (pair[0].astype(np.single) - blurred_high.astype(np.single)) / 255
    high = cv2.normalize(high, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
    high_hy = high.astype(np.single) / 255
    blurred_low_hy = blurred_low.astype(np.single) / 255
    hybrid = (0.5 * blurred_low_hy + 0.5 * high_hy) * 255
    hybrid = hybrid.astype(np.uint8)
    return blurred_low, high, hybrid

def log_mag_fft(image, boolean):
    if boolean:
        gr = image
    else:
        gr = gray(image)
    out = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gr))))
    return out

boolean = False
save = "../output/"
# afghan_pair = create_pair(afghan_before, afghan_after, boolean)
afghan_pair = check_pair(getImages(dir + pair1), boolean)
afghan_pair.append("Afghan_Pair")
# bike_pair = create_pair(bicycle, motorcycle, boolean)
bike_pair = check_pair(getImages(dir + pair2), boolean)
bike_pair.append("Bike_Pair")
# flying_pair = create_pair(bird, plane, boolean)
flying_pair = check_pair(getImages(dir + pair3), boolean)
flying_pair.append("Flying_Pair")
# animal_pair = create_pair(cat, dog, boolean)
animal_pair = check_pair(getImages(dir + pair4), boolean)
animal_pair.append("Animal_Pair")
# face_pair = create_pair(einstein, marilyn, boolean)
face_pair = check_pair(getImages(dir + pair5), boolean)
face_pair.append("Face_Pair")
# sea_pair = create_pair(fish, submarine, boolean)
sea_pair = check_pair(getImages(dir + pair6), boolean)
sea_pair.append("Sea_Pair")
# makeup_pair = create_pair(makeup_before, makeup_after, boolean)
makeup_pair = check_pair(getImages(dir + pair7), boolean)
makeup_pair.append("Makeup_Pair")
pairs = [afghan_pair, bike_pair, flying_pair, animal_pair, face_pair, sea_pair, makeup_pair]
for i in range(len(pairs)):
    low, high, hybrid = getHybrid(pairs[i])
    string_val = pairs[i][2]
    # folder = string_val + '/'
    final_image = np.hstack((low, high, hybrid))
    fft = log_mag_fft(hybrid, boolean)
    cv2.imshow("Low Pass", low)
    cv2.imshow("High Pass", high)
    cv2.imshow("Hybrid", hybrid)
    plt.imshow(fft, cmap='gray')
    plt.show()
    cv2.imwrite(save + string_val + '.jpg', final_image)
    plt.imsave(save + string_val + '_FFT.jpg', fft, cmap='gray')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
