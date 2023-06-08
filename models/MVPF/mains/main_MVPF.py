import os,sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from libs import para, main_common
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = para.MVPF_main(parser)
    parser = para.common_para(parser)
    parser = para.MVPF(parser)
    args = parser.parse_args()
    main_common.main(args)
