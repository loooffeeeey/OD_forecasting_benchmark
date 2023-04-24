# Conversion of Washington DC Taxi Trips (2017): https://www.kaggle.com/bvc5283/dc-taxi-trips
import argparse
import pandas as pd
import numpy as np
MIN_LAT = 38.7919   # Exclusive
MAX_LAT = 38.9960   # Inclusive
MIN_LNG = -77.1200  # Inclusive
MAX_LNG = -76.9093  # Exclusive

def convertData(inFile, outFile, startDate, endDate):
    df = pd.read_csv(inFile)
    df_out = df[['StartDateTime',
                 'OriginLatitude', 'OriginLongitude',
                 'DestinationLatitude', 'DestinationLongitude']].rename(columns={
        'StartDateTime': 'request time',
        'OriginLatitude': 'src lat',
        'OriginLongitude': 'src lng',
        'DestinationLatitude': 'dst lat',
        'DestinationLongitude': 'dst lng'
    })
    df_out['volume'] = 1
    # Select a day of data
    df_out['request time'] = pd.to_datetime(df_out['request time'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    df_out.dropna(subset=['request time'], inplace=True)  # Drop rows with invalid date format
    df_out['request time'] = df_out['request time'].dt.tz_localize(None)
    df_out['request time'] = df_out['request time'].dt.strftime("%Y-%m-%d %H:%M:%S")
    mask = ((df_out['request time'] >= startDate) & (df_out['request time'] < endDate)).values
    df_out = df_out.iloc[mask]
    # Filter abnormal data: In Washington DC, any coordinate as 0 is abnormal
    df_out.dropna(subset=['src lat', 'src lng', 'dst lat', 'dst lng'], inplace=True)
    mask = ((df_out['src lat'] * df_out['src lng'] * df_out['dst lat'] * df_out['dst lng'] != 0) &
            (df_out['src lat'] > MIN_LAT) & (df_out['src lat'] <= MAX_LAT) &
            (df_out['dst lat'] > MIN_LAT) & (df_out['dst lat'] <= MAX_LAT) &
            (df_out['src lng'] >= MIN_LNG) & (df_out['src lng'] < MAX_LNG) &
            (df_out['dst lng'] >= MIN_LNG) & (df_out['dst lng'] < MAX_LNG)).values
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
        python dc2017.py -i taxi_final.csv -o dc2017.csv -sd 2017-06-18 -ed 2017-06-25
    """
    # Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='taxi_final.csv',
                        help='The input data file to be converted, default={}'.format('taxi_final.csv'))
    parser.add_argument('-o', '--output', type=str, default='dc2017.csv',
                        help='The output converted data file, default={}'.format('ny2016.csv'))
    parser.add_argument('-sd', '--startDate', type=str, default='2017-06-18',
                        help='The start date to filter (inclusive), default={}'.format('2017-06-18'))
    parser.add_argument('-ed', '--endDate', type=str, default='2017-06-25',
                        help='The end date to filter (exclusive), default={}'.format('2017-06-25'))
    FLAGS, unparsed = parser.parse_known_args()

    convertData(FLAGS.input, FLAGS.output, FLAGS.startDate, FLAGS.endDate)
