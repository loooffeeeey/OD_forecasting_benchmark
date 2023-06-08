"""
Analyze the data on various aspects
"""
import argparse
import os
from PIL import Image
import json
import numpy as np
EDGE_COL = (0, 0, 0)    # Default: Black
EDGE_HALF_WIDTH = 2     # EDGE_HALF_WIDTH + 1 + EDGE_HALF_WIDTH, Default: 2 + 1 + 2 = 5


def displayGrid(pic_path, gridInfo_path):
    gridInfo = json.load(open(gridInfo_path))
    latGridNum, lngGridNum = gridInfo['latGridNum'], gridInfo['lngGridNum']

    # x is lng, y is lat, top left is origin
    img = Image.open(pic_path)
    img = img.convert("RGBA")
    data = np.array(img)

    xDim, yDim = data.shape[0], data.shape[1]
    xStep = xDim / lngGridNum
    yStep = yDim / latGridNum
    xAx = [int(i * xStep) for i in range(lngGridNum + 1)]
    yAx = [int(i * yStep) for i in range(latGridNum + 1)]

    data = renderVerL(data, xAx)
    data = renderHorL(data, yAx)

    img_display = Image.fromarray(data)
    img_display.save('grid_map.png')
    print('grid_map.png saved to local.')
    img_display.show()


def renderHorL(data, yAxis):
    for ax in yAxis:
        for i in range(data.shape[0]):
            data = renderCol(data, i, ax)
            for w in range(EDGE_HALF_WIDTH):
                data = renderCol(data, i, ax - w)
                data = renderCol(data, i, ax + w)
    return data


def renderVerL(data, xAxis):
    for ax in xAxis:
        for i in range(data.shape[1]):
            data = renderCol(data, ax, i)
            for w in range(EDGE_HALF_WIDTH):
                data = renderCol(data, ax - w, i)
                data = renderCol(data, ax + w, i)
    return data


def renderCol(data, x, y):
    if 0 <= x < data.shape[0] and 0 <= y < data.shape[1]:
        data[x][y][0:3] = EDGE_COL
    return data


if __name__ == '__main__':
    """
        Usage Example:
            python GridVisualizer.py -p New_York_Map.png -g grid_info.json
    """
    # Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pic', type=str, default='New_York_Map.png',
                        help='The input image file to be visualized, default={}'.format('New_York_Map.png'))
    parser.add_argument('-g', '--gridInfo', type=str, default='grid_info.json',
                        help='The json file storing the grid information, default={}'.format('grid_info.json'))
    FLAGS, unparsed = parser.parse_known_args()

    # DO STH
    picName = FLAGS.pic
    gridInfoName = FLAGS.gridInfo
    if os.path.isfile(picName) and os.path.isfile(gridInfoName):
        displayGrid(picName, gridInfoName)
