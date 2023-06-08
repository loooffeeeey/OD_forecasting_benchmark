"""
Concatenate the data frames
"""
import argparse
import pandas as pd
import os


def concatenate(data, outFile):
    frames = []
    for f in data:
        df = pd.read_csv(f)
        frames.append(df)
    concatenated = pd.concat(frames)    # Concatenate
    concatenated['request time'] = pd.to_datetime(concatenated['request time'])
    concatenated.sort_values(by=['request time'], inplace=True)
    concatenated.to_csv(outFile, index=False)   # Store


if __name__ == '__main__':
    """
        Usage Example:
            python DataConcatenation.py -d data0.csv data1.csv -o data_out.csv
    """
    # Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, nargs='*',
                        help='The input data files to be concatenated, multiple data frames supported')
    parser.add_argument('-o', '--output', type=str, default='concatenated.csv',
                        help='The output concatenated data file, default = {}'.format('concatenated.csv'))
    FLAGS, unparsed = parser.parse_known_args()

    # DO STH
    if FLAGS.data is None:
        print('No data is specified!')
        exit(-1)
    for f in FLAGS.data:
        if not os.path.isfile(f):
            print('{} is not a valid file!'.format(f))
            exit(-2)

    concatenate(FLAGS.data, FLAGS.output)
