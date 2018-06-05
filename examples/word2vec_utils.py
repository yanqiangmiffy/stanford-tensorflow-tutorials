from collections import Counter
import random
import os
import sys
sys.path.append('..')
import zipfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import utils

def read_data(file_path):
    """
        将tokens读取到list里
        应该有17005207个tokens
    """
    with zipfile.ZipFile(file_path) as f:
        # text8.zip只有一个文件，所以是namelist()[0]
        # tensorflow.compat.as_str  将 bytes 和 unicode 类型的字符串统一转化为  unicode 字符串.
        words = tf.compat.as_str(f.read(f.namelist()[0])).split()
        return words

def build_vocab(words, vocab_size, visual_fld):
    pass
if __name__ == '__main__':
    local_dest = 'data/text8.zip'
    download_url = 'http://mattmahoney.net/dc/text8.zip'
    expected_byte = 31344016
    utils.download_one_file(download_url, local_dest, expected_byte)
    words = read_data(local_dest)
    print(words)
