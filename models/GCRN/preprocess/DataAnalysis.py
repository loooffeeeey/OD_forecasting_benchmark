"""
Analyze the data on various aspects
"""
import argparse
import pandas as pd
import os


def statistics(file):
    df = pd.read_csv(file)
    df['request time'] = pd.to_datetime(df['request time'])

    nReq = df['volume'].sum()
    print('Total Requests: {}'.format(nReq))

    minT, maxT = df['request time'].min(), df['request time'].max()
    tSpan = (maxT - minT) / pd.Timedelta(hours=1)
    print('Time span: [{}, {}] => {} hours'.format(minT, maxT, tSpan))

    print('Average requests per hour = {}'.format(nReq / tSpan))

    minLat, maxLat = df[['src lat', 'dst lat']].min().min(), df[['src lat', 'dst lat']].max().max()
    minLng, maxLng = df[['src lng', 'dst lng']].min().min(), df[['src lng', 'dst lng']].max().max()
    print('Latitude ∈ [{}, {}]'.format(minLat, maxLat))
    print('Longitude ∈ [{}, {}]'.format(minLng, maxLng))

    datetimes = df['request time'].dt
    maxVol = df.groupby([datetimes.year, datetimes.month, datetimes.day, datetimes.hour]).sum()['volume'].max()
    print('Maximum Volume per hour = {}'.format(maxVol))


if __name__ == '__main__':
    """
        Usage Example:
            python DataAnalysis.py -d ny2016_0101to0331.csv
    """
    # Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='ny2016_0101to0331.csv',
                        help='The input data file to be analyzed, default={}'.format('ny2016_0101to0331.csv'))
    FLAGS, unparsed = parser.parse_known_args()

    # DO STH
    fileName = FLAGS.data
    if os.path.isfile(fileName):
        statistics(fileName)
