# Conversion of Peru Uber Dataset (2010): https://www.kaggle.com/marcusrb/uber-peru-dataset
import argparse
import pandas as pd
import numpy as np


def convertData(inFile, outFile):
    df = pd.read_csv(inFile, sep=';')
    df_out = df[['start_at',
                 'start_lat', 'start_lon',
                 'end_lat', 'end_lon']].rename(columns={
        'start_at': 'request time',
        'start_lat': 'src lat',
        'start_lon': 'src lng',
        'end_lat': 'dst lat',
        'end_lon': 'dst lng'
    })
    df_out['volume'] = 1
    df_out['request time'] = pd.to_datetime(df_out['request time'])
    # Transform coordinate from -12,13 to -12.13
    df_out['src lat'] = df_out['src lat'].apply(lambda x: float(x.replace(',', '.')))
    df_out['src lng'] = df_out['src lng'].apply(lambda x: float(x.replace(',', '.')))
    df_out['dst lat'] = df_out['dst lat'].apply(lambda x: float(x.replace(',', '.')))
    df_out['dst lng'] = df_out['dst lng'].apply(lambda x: float(x.replace(',', '.')))
    # Filter abnormal data: In Peru, any coordinate as 0 is abnormal
    df_out.dropna(subset=['src lat', 'src lng', 'dst lat', 'dst lng'], inplace=True)
    mask = ((df_out['src lat'] * df_out['src lng'] * df_out['dst lat'] * df_out['dst lng'] != 0) &
            (df_out['src lat'] >= -90) & (df_out['src lat'] <= 90) &
            (df_out['dst lat'] >= -90) & (df_out['dst lat'] <= 90) &
            (df_out['src lng'] >= -180) & (df_out['src lng'] <= 180) &
            (df_out['dst lng'] >= -180) & (df_out['dst lng'] <= 180)).values
    df_out = df_out.iloc[mask]
    # Sort by Date
    df_out.sort_values(by=['request time'], inplace=True)
    missingStat(df_out)
    df_out.to_csv(outFile, index=False)


def missingStat(df):
    nullSheet = df.isnull().sum()
    nNull = np.sum(nullSheet)
    total = np.prod(df.shape)
    print('\nMissing Count:')
    print(nullSheet)
    print('Missing Percentage = %.2f / %.2f = %.2f%%\n' % (float(nNull), float(total), nNull / total * 100))


if __name__ == '__main__':
    """
    Usage Example:
        python peru2010.py -i uber_peru_2010.csv -o peru2010.csv
    """
    # Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='uber_peru_2010.csv',
                        help='The input data file to be converted, default={}'.format('uber_peru_2010.csv'))
    parser.add_argument('-o', '--output', type=str, default='peru2010.csv',
                        help='The output converted data file, default={}'.format('peru2010.csv'))
    FLAGS, unparsed = parser.parse_known_args()

    convertData(FLAGS.input, FLAGS.output)
