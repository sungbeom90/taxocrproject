import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import fnmatch
import cv2
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


def encode_to_labels(char_list, txt):
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            pass

    return dig_lst


def data_load(path, mode, char_list):
    file_name_list = []

    h_size = 32
    w_size = 256

    max_label_len = 63

    if mode == 1:
        print('Test Data loading...')

        test_img = []
        test_txt = []
        test_input_length = []
        test_label_length = []
        test_orig_txt = []

    i = 1
    flag = 0

    exceptional_class = {r'$$e1$$': r'￦', r'$$e2$$': '/', r'$$e3$$': ':', r'$$e4$$': '*', r'$$e5$$': '?',
                         r'$$e6$$': '"', r'$$7e7$$': '<',  r'$$e8$$': '>', r'$$e9$$': '|', r'$$e10$$': '_'}
    exceptional_class_keys = list(exceptional_class.keys())

    h_class = {r'$$h1$$': '外'}
    h_class_keys = list(h_class.keys())

    for root, dirnames, filenames in os.walk(path):

        for f_name in fnmatch.filter(filenames, '*.jpg'):
            file_name_list.append(f_name)

            fn = os.path.join(root, f_name)
            stream = open(fn, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

            if len(bgrImage.shape) == 3:
                img = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2GRAY)
            else:
                img = bgrImage

            h, w = img.shape

            if w > w_size:
                img = cv2.resize(img, dsize=(w_size, h), interpolation=cv2.INTER_AREA)
                w = w_size

            if h > h_size:
                img = cv2.resize(img, dsize=(w, h_size), interpolation=cv2.INTER_AREA)
                h = h_size

            if w < w_size:
                img = cv2.resize(img, dsize=(w_size, h), interpolation=cv2.INTER_AREA)
                w = w_size

            if h < h_size:
                img = cv2.resize(img, dsize=(w, h_size), interpolation=cv2.INTER_AREA)
                h = h_size

            img = np.expand_dims(img, axis=2)

            img = img / 255.

            txt = f_name.split('_')[1]

            for except_index in range(len(exceptional_class)):
                if exceptional_class_keys[except_index] in txt:
                    txt = txt.replace(exceptional_class_keys[except_index],
                                      exceptional_class[exceptional_class_keys[except_index]])

            for except_index in range(len(h_class)):
                if h_class_keys[except_index] in txt:
                    txt = txt.replace(h_class_keys[except_index],
                                      h_class[h_class_keys[except_index]])

            if mode == 1:
                test_orig_txt.append(txt)
                test_label_length.append(len(txt))
                test_input_length.append(max_label_len)
                test_img.append(img)
                test_txt.append(encode_to_labels(char_list, txt))
                if i % 100 == 0:
                    print('Test Data loading index: ', i)

            i += 1

    if mode == 1:
        test_img = np.array(test_img)

        test_padded_txt = pad_sequences(test_txt, maxlen=max_label_len, padding='post', value=len(char_list))

        return test_img, test_padded_txt