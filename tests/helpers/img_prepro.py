from PIL import Image
import numpy as np
import cv2


def load_single_img_resize(image_route, width: int, height: int):
    image_data = []
    im = Image.open(image_route)
    im = im.convert("L")
    im = detection_preprocess(im)

    img_width, img_height = im.size

    read = np.array(im.resize((width, height)), np.float32) / 255

    ratio = (float(width / 2 / img_width), float(height / 2 / img_height))

    size_data = ratio

    img_arr = np.ndarray((width, height, 1), np.float32)

    pads = ((0, 0), (0, 0))

    for i in [0]:
        x = read[:, :]
        pad = np.pad(x, pads, "constant", constant_values=1)
        pad = np.resize(pad, (width, height))
        img_arr[:, :, 0] = pad

    image_data.append(img_arr)

    return img_arr, size_data


def detection_preprocess(image: Image):
    bgrImage = np.array(image, dtype=np.uint8)
    print(bgrImage.shape)

    gray = bgrImage

    b_w = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    cv2.imshow("THRESH_BINARY_INV", b_w)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, _ = cv2.findContours(b_w, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(b_w, contours, -1, (255), 3)

    new_img = b_w

    h, w = new_img.shape[:2]

    horizontal_img = new_img
    vertical_img = new_img

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    horizontal_img = cv2.erode(horizontal_img, horizontal_kernel, iterations=1)

    cv2.imshow("y100 침식", horizontal_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    horizontal_img = cv2.dilate(horizontal_img, horizontal_kernel, iterations=1)

    cv2.imshow("y100 팽창", horizontal_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    vertical_img = cv2.erode(vertical_img, vertical_kernel, iterations=1)

    cv2.imshow("x100 침식", vertical_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    vertical_img = cv2.dilate(vertical_img, vertical_kernel, iterations=1)

    cv2.imshow("x100 팽창", vertical_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    mask_img = horizontal_img + vertical_img

    cv2.imshow("이미지 브로드케스팅", mask_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    b_w2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    cv2.imshow("이미지 THRESH_BINARY", b_w2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    re_img = np.bitwise_or(b_w2, mask_img)

    return Image.fromarray(re_img)
