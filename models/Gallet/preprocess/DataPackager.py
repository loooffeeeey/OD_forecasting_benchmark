"""
Package the data into a specified folder, which will contain:
1. grid_info.json (minLat, maxLat, minLng, maxLng, girdH, gridW, latLen, latGridNum, gridLat, ..., gridNum)
2. GeoGraph.dgl which stores the geographical graph GeoGraph
3. Passenger Request Data separated in hours ((1/request.csv, 1/GDVQ.npy, 1/FBGraphs.dgl), ...)
"""
import argparse
import os
import sys
import math
import json
import numpy as np
import pandas as pd
import torch
import multiprocessing

from tqdm import tqdm

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import dgl
sys.stderr.close()
sys.stderr = stderr

EPSILON = 1e-12
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Features
DAY_OF_WEEK = ['weekday', 'weekend']
PERIOD_OF_DAY = ['morning', 'noon', 'afternoon', 'night', 'midnight']


def decideDayOfWeek(dayOfWeek):
    return 'weekday' if dayOfWeek < 5 else 'weekend'


def decidePeriodOfDay(hourOfDay):
    if hourOfDay < 6:
        return 'midnight'
    elif hourOfDay < 12:
        return 'morning'
    elif hourOfDay < 14:
        return 'noon'
    elif hourOfDay < 18:
        return 'afternoon'
    else:   # 18 ~ 23
        return 'night'


def haversine(c0, c1):
    """
    :param c0: coordinate 0 in form (lat0, lng0) with degree as unit
    :param c1: coordinate 1 in form (lat1, lng1) with degree as unit
    :return: The haversine distance of c0 and c1 in km
    Compute the haversine distance between
    https://en.wikipedia.org/wiki/Haversine_formula
    """
    dLat = math.radians(c1[0] - c0[0])
    dLng = math.radians(c1[1] - c0[1])
    lat0 = math.radians(c0[0])
    lat1 = math.radians(c1[0])
    form0 = math.pow(math.sin(dLat / 2), 2)
    form1 = math.cos(lat0) * math.cos(lat1) * math.pow(math.sin(dLng / 2), 2)
    radius_of_earth = 6371  # km
    dist = 2 * radius_of_earth * math.asin(math.sqrt(form0 + form1))
    return dist


def path2FileNameWithoutExt(path):
    """
    get file name without extension from path
    :param path: file path
    :return: file name without extension
    """
    return os.path.splitext(path)[0]


def getGridInfo(minLat, maxLat, minLng, maxLng, refGridW=2.5, refGridH=2.5):
    """
    :param minLat: lower boundary of region's latitude
    :param maxLat: upper boundary of region's latitude
    :param minLng: lower boundary of region's longitude
    :param maxLng: upper boundary of region's longitude
    :param refGridW: reference width of a grid in km, will auto-adjust later
    :param refGridH: reference height of a grid in km, will auto-adjust later
    :return: grid_info dictionary
    """
    grid_info = {
        'minLat': minLat,
        'maxLat': maxLat,
        'minLng': minLng,
        'maxLng': maxLng
    }
    grid_info['latLen'] = haversine((grid_info['minLat'], grid_info['maxLng']),
                                    (grid_info['maxLat'], grid_info['maxLng']))
    grid_info['latGridNum'] = round(grid_info['latLen'] / refGridH)
    grid_info['gridH'] = grid_info['latLen'] / grid_info['latGridNum']
    grid_info['gridLat'] = (grid_info['maxLat'] - grid_info['minLat']) / grid_info['latGridNum']

    grid_info['lngLen'] = haversine((grid_info['maxLat'], grid_info['minLng']),
                                    (grid_info['maxLat'], grid_info['maxLng']))
    grid_info['lngGridNum'] = round(grid_info['lngLen'] / refGridW)
    grid_info['gridW'] = grid_info['lngLen'] / grid_info['lngGridNum']
    grid_info['gridLng'] = (grid_info['maxLng'] - grid_info['minLng']) / grid_info['lngGridNum']

    grid_info['gridNum'] = grid_info['latGridNum'] * grid_info['lngGridNum']
    print('-> Grid info retrieved.')
    return grid_info


def saveGridInfo(grid_info, fPath):
    with open(fPath, 'w') as f:
        json.dump(grid_info, f)
    print('-> grid_info saved to {}'.format(fPath))


def makeGridNodes(grid_info):
    """
    Generate a grid node coordinate list where each grid represents a node
    :param grid_info: Grid Map Information
    :return: A Grid Coordinate List st. in each position (Grid ID) a (latitude, longitude) pair is stored
    """
    leftLng = grid_info['minLng'] + grid_info['gridLng'] / 2
    midLat = grid_info['maxLat'] - grid_info['gridLat'] / 2
    grid_nodes = []
    for i in range(grid_info['latGridNum']):
        midLng = leftLng
        for j in range(grid_info['lngGridNum']):
            grid_nodes.append((midLat, midLng))
            midLng += grid_info['gridLng']
        midLat -= grid_info['gridLat']
    print('-> Grid nodes generated.')
    return grid_nodes


def pushGraphEdge(gSrc: list, gDst: list, wList, src, dst, weight):
    gSrc.append(src)
    gDst.append(dst)
    if wList is not None and weight is not None:
        wList.append(weight)
        return gSrc, gDst, wList
    else:
        return gSrc, gDst


def matOD2G(mat, oList: list, dList: list, nGNodes, hasPreWeights=True):
    # Create DGL Graph
    graph = dgl.graph((oList, dList), num_nodes=nGNodes)

    if hasPreWeights:
        # pre weights
        matSum = np.sum(mat, axis=0)
        for nj in range(nGNodes):
            if matSum[nj] == 0:
                continue
            for ni in range(nGNodes):
                mat[ni][nj] /= matSum[nj]

        # Transform node data mat to edge data edges
        edges = []
        for i in range(len(oList)):
            edges.append([mat[oList[i]][dList[i]]])

        graph.edata['pre_w'] = torch.Tensor(edges)

    return graph


def getGeoGraph(grid_nodes, L):
    """
    Get RTc data object with the given information
    :param grid_nodes: Grid Coordinate List storing the coordinates of each grid/node
    :param L: Threshold distance for geographical neighborhood decision making
    :return: Geographical Neighborhood DGL Graph
    """
    adjacency_matrix = np.zeros((len(grid_nodes), len(grid_nodes)))
    RSrcList, RDstList = [], []
    TcMat = np.zeros((len(grid_nodes), len(grid_nodes)))
    for i in range(len(grid_nodes)):
        for j in range(len(grid_nodes)):
            adjacency_matrix[i][j] = haversine(grid_nodes[i], grid_nodes[j])
            if i != j and adjacency_matrix[i][j] <= L:
                # if i->j is small enough, j is i's geographical neighbor, j should propagate its features to i, so j->i
                RSrcList, RDstList = pushGraphEdge(RSrcList, RDstList, None, j, i, None)
                TcMat[j][i] = 1 / (adjacency_matrix[i][j] + EPSILON)

    GeoGraph = matOD2G(mat=TcMat, oList=RSrcList, dList=RDstList, nGNodes=len(grid_nodes), hasPreWeights=True)

    print('-> Geographical info generated.')
    return GeoGraph


def saveGeoGraph(geoG, fPath):
    dgl.save_graphs(fPath, geoG)
    print('-> Geographical info saved to {}'.format(fPath))


def inWhichGrid(coord, grid_info):
    """
    Specify which grid it is in for a given coordinate
    :param coord: (latitude, longitude)
    :param grid_info: grid_info dictionary
    :return: row, column, grid ID
    """
    lat, lng = coord
    row = math.floor((grid_info['maxLat'] - lat) / grid_info['gridLat'])
    col = math.floor((lng - grid_info['minLng']) / grid_info['gridLng'])
    gridID = row * grid_info['lngGridNum'] + col
    return row, col, gridID


def constructReqMat(df, grid_info):
    request_matrix = np.zeros((grid_info['gridNum'], grid_info['gridNum']))
    for df_i in range(len(df)):
        cur_data = df.iloc[df_i]
        src_row, src_col, src_id = inWhichGrid((cur_data['src lat'], cur_data['src lng']), grid_info)
        dst_row, dst_col, dst_id = inWhichGrid((cur_data['dst lat'], cur_data['dst lng']), grid_info)
        request_matrix[src_id][dst_id] += 1
    return request_matrix.astype(np.float32)


def ID2Coord(gridID, grid_info):
    """
    Given a grid ID, decide the coordinate of that grid
    :param gridID: Given Grid ID
    :param grid_info: Grid map information
    :return: Grid Map Coordinates of that grid in format of (row, col)
    """
    row = math.floor(gridID / grid_info['lngGridNum'])
    col = gridID - row * grid_info['lngGridNum']
    return row, col


def oneHotEncode(val, valList: list):
    return [1 if val == valInList else 0 for valInList in valList]


def constructFBGraph(request_matrix, num_grid_nodes, mix=False):
    if mix:     # Forward & Backward mixed together (For GEML)
        P_src_list, P_dst_list = [], []
        P_mat = np.zeros((num_grid_nodes, num_grid_nodes))

        for rmi in range(num_grid_nodes):
            for rmj in range(num_grid_nodes):
                # rmi -> rmj, rmj is rmi's forward neighbor, rmi is rmj's backward neighbor
                # Forward Neighborhood: features of rmj should propagate to rmi, thus rmj->rmi
                # Backward Neighborhood: features of rmi should propagate to rmj, thus rmi->rmj
                # Mixed Neighborhood: rmj is rmi's neighbor either rmi->rmj or rmj->rmi
                if request_matrix[rmi][rmj] > 0 or request_matrix[rmj][rmi] > 0:
                    P_src_list, P_dst_list = pushGraphEdge(P_src_list, P_dst_list, None, rmj, rmi, None)
                    P_mat[rmj][rmi] = request_matrix[rmi][rmj] + request_matrix[rmj][rmi]

        FBN_graph = matOD2G(mat=P_mat, oList=P_src_list, dList=P_dst_list, nGNodes=num_grid_nodes, hasPreWeights=True)
        return FBN_graph
    else:
        Pa_src_list, Pa_dst_list = [], []   # Psi & a
        Pb_src_list, Pb_dst_list = [], []   # Phi & b
        Pa_mat = np.zeros((num_grid_nodes, num_grid_nodes))
        Pb_mat = np.zeros((num_grid_nodes, num_grid_nodes))

        for rmi in range(num_grid_nodes):
            for rmj in range(num_grid_nodes):
                # rmi -> rmj, rmj is rmi's forward neighbor, rmi is rmj's backward neighbor
                # Forward Neighborhood: features of rmj should propagate to rmi, thus rmj->rmi
                # Backward Neighborhood: features of rmi should propagate to rmj, thus rmi->rmj
                # Mixed Neighborhood: rmj is rmi's neighbor either rmi->rmj or rmj->rmi
                if request_matrix[rmi][rmj] > 0:
                    Pa_src_list, Pa_dst_list = pushGraphEdge(Pa_src_list, Pa_dst_list, None, rmj, rmi, None)
                    Pa_mat[rmj][rmi] = request_matrix[rmi][rmj]
                    Pb_src_list, Pb_dst_list = pushGraphEdge(Pb_src_list, Pb_dst_list, None, rmi, rmj, None)
                    Pb_mat[rmi][rmj] = request_matrix[rmi][rmj]

        FN_graph = matOD2G(mat=Pa_mat, oList=Pa_src_list, dList=Pa_dst_list, nGNodes=num_grid_nodes, hasPreWeights=True)
        BN_graph = matOD2G(mat=Pb_mat, oList=Pb_src_list, dList=Pb_dst_list, nGNodes=num_grid_nodes, hasPreWeights=True)
        return FN_graph, BN_graph


def handleRequestData(i, totalH, folder, lowT, df_split, export_requests, grid_info):
    curH = i + 1
    # print('-> Splitting hour-wise data No.{}/{}.'.format(curH, totalH))

    num_grid_nodes = grid_info['gridNum']

    # Folder for this split of data
    curDir = os.path.join(folder, str(curH))
    if not os.path.isdir(curDir):
        os.mkdir(curDir)

    dayOfWeek = lowT.weekday()  # Mon: 0, ..., Sun: 6
    dayTypeOfWeek = decideDayOfWeek(dayOfWeek)
    oneHotDOW = [1 if j == dayOfWeek else 0 for j in range(7)]
    oneHotDTOW = oneHotEncode(dayTypeOfWeek, DAY_OF_WEEK)

    hourOfDay = lowT.hour       # 0 ~ 23
    periodOfDay = decidePeriodOfDay(hourOfDay)
    oneHotHOD = [1 if j == hourOfDay else 0 for j in range(24)]
    oneHotPOD = oneHotEncode(periodOfDay, PERIOD_OF_DAY)

    GDVQ = {}

    # Save request.csv
    if export_requests:
        df_split.to_csv(os.path.join(curDir, 'request.csv'), index=False)

    # Get request matrix G
    request_matrix = constructReqMat(df_split, grid_info)
    GDVQ['G'] = request_matrix.astype(np.float32)

    # Get Feature Matrix V
    inDs = np.sum(request_matrix, axis=0)  # Col-wise: Total number of nodes pointing to current node = In Degree
    outDs = np.sum(request_matrix, axis=1)  # Row-wise: Total number of nodes current node points to = Out Degree
    GDVQ['D'] = outDs.astype(np.float32)

    # for further calculations
    GDVQ['inD_min'] = np.min(inDs.astype(np.float32))
    GDVQ['inD_max'] = np.max(inDs.astype(np.float32))
    GDVQ['outD_min'] = np.min(outDs.astype(np.float32))
    GDVQ['outD_max'] = np.max(outDs.astype(np.float32))

    feature_vectors = []
    query_feature_vectors = []
    for vi in range(num_grid_nodes):
        viRow, viCol = ID2Coord(vi, grid_info)
        # query vector
        query_feature_vector = oneHotDOW + oneHotDTOW + oneHotHOD + oneHotPOD + [
            viRow / (grid_info['latGridNum'] - 1),
            viCol / (grid_info['lngGridNum'] - 1),
            vi / (num_grid_nodes - 1)
        ]
        query_feature_vectors.append(query_feature_vector)
        # feature vector: Note that Ds are not yet normalized
        feature_vector = [
            outDs[vi],
            inDs[vi]
        ] + query_feature_vector
        feature_vectors.append(feature_vector)
    feature_matrix = np.array(feature_vectors)
    query_feature_matrix = np.array(query_feature_vectors)
    GDVQ['V'] = feature_matrix.astype(np.float32)
    GDVQ['Q'] = query_feature_matrix.astype(np.float32)

    # Save GDVQ as GDVQ.npy
    np.save(os.path.join(curDir, 'GDVQ.npy'), GDVQ)

    # Get Psi (Forward Neighborhood) and Phi (Backward Neighborhood)
    FNGraph, BNGraph = constructFBGraph(request_matrix, num_grid_nodes, mix=False)
    dgl.save_graphs(os.path.join(curDir, 'FBGraphs.dgl'), [FNGraph, BNGraph])

    mixFBNGraph = constructFBGraph(request_matrix, num_grid_nodes, mix=True)
    dgl.save_graphs(os.path.join(curDir, 'FBGraphMix.dgl'), mixFBNGraph)

    return request_matrix


def minMaxScale(x, minVal, maxVal):
    if x < minVal or x > maxVal:
        sys.stderr.write('MinMaxScaling: %.2f not in [%.2f, %.2f]!\n' % (x, minVal, maxVal))
        exit(-6)
    if maxVal - minVal == 0:
        sys.stderr.write('MinMaxScaling: Warning --> min(%.2f) == max(%.2f)\n' % (minVal, maxVal))
        return 0
    return (x - minVal) / (maxVal - minVal)


def normDnV(i, totalH, folder, inD_min, inD_max, outD_min, outD_max):
    curH = i + 1
    # print('-> Normalizing Ds in Vs for data No.{}/{}.'.format(curH, totalH))
    GDVQ = np.load(os.path.join(folder, str(curH), 'GDVQ.npy'), allow_pickle=True).item()
    curV = GDVQ['V']
    for ni in range(len(curV)):
        curV[ni][0] = minMaxScale(curV[ni][0], outD_min, outD_max)
        curV[ni][1] = minMaxScale(curV[ni][1], inD_min, inD_max)
    GDVQ['V'] = curV
    np.save(os.path.join(folder, str(curH), 'GDVQ.npy'), GDVQ)


def splitData(fPath, folder, grid_nodes, grid_info, export_requests=1, num_workers=10):
    """
    Split data in hours (request.csv, GDVQ.npy) of each DDW Snapshot Graph
    :param num_workers: number of cpu cores to split data asynchronously
    :param fPath: The path of request data file
    :param folder: The path of the working directory/folder
    :param grid_nodes: Grid Coordinate List storing the coordinates of each grid/node
    :param grid_info: Grid Map Information
    :param export_requests: whether the split requests should be exported if space is enough
    :return: nothing
    """
    df = pd.read_csv(fPath)
    df['request time'] = pd.to_datetime(df['request time'])
    minT, maxT = df['request time'].min(), df['request time'].max()
    totalH = round((maxT - minT) / pd.Timedelta(hours=1))
    lowT, upT = minT, minT + pd.Timedelta(hours=1)
    print('-> Dataframe prepared. Total hours = %s.' % totalH)

    # Save req_info
    req_info = {
        'name': path2FileNameWithoutExt(fPath),
        'minT': minT.strftime(DATE_FORMAT),
        'maxT': maxT.strftime(DATE_FORMAT),
        'totalH': totalH
    }
    req_info_path = os.path.join(folder, 'req_info.json')
    with open(req_info_path, 'w') as f:
        json.dump(req_info, f)
    print('-> requests info saved to %s' % req_info_path)

    request_matrices = []

    # Split hour-wise data
    print('-> Splitting %d hour-wise data.' % totalH)
    pool = multiprocessing.Pool(processes=num_workers)
    pbar_req_data_tasks = tqdm(total=totalH)

    def pbar_req_data_tasks_update(res):
        request_matrices.append(res)
        pbar_req_data_tasks.update()

    for i in range(totalH):
        # Filter data
        mask = ((df['request time'] >= lowT) & (df['request time'] < upT)).values
        df_split = df.iloc[mask]

        # Handle data
        pool.apply_async(handleRequestData, args=(i, totalH, folder, lowT, df_split, export_requests, grid_info), callback=pbar_req_data_tasks_update)
        # handleRequestData(i, totalH, folder, lowT, df_split, export_requests, grid_info)    # DEBUG

        lowT += pd.Timedelta(hours=1)
        upT += pd.Timedelta(hours=1)

    pool.close()
    pool.join()
    pbar_req_data_tasks.close()

    # Normalize Ds in the feature vectors
    # 1. Get min max
    inD_min, outD_min = float('inf'), float('inf')
    inD_max, outD_max = -1, -1
    print('\n-> Scanning %d data to calculate min & max.' % totalH)
    for i in tqdm(range(totalH)):
        curH = i + 1
        GDVQ = np.load(os.path.join(folder, str(curH), 'GDVQ.npy'), allow_pickle=True).item()
        inD_min = min(GDVQ['inD_min'], inD_min)
        outD_min = min(GDVQ['outD_min'], outD_min)
        inD_max = max(GDVQ['inD_max'], inD_max)
        outD_max = max(GDVQ['outD_max'], outD_max)
        del GDVQ['inD_min']
        del GDVQ['inD_max']
        del GDVQ['outD_min']
        del GDVQ['outD_max']
        np.save(os.path.join(folder, str(curH), 'GDVQ.npy'), GDVQ)
    print("\ninD_min = %.2f, inD_max = %.2f\noutD_min = %.2f, outD_max = %.2f" % (
        inD_min, inD_max, outD_min, outD_max
    ))
    # 2. Normalize Ds in Vs
    print('-> Normalizing Ds in Vs for %d data.' % totalH)
    pool = multiprocessing.Pool(processes=num_workers)
    pbar_norm_tasks = tqdm(total=totalH)

    def pbar_norm_tasks_update(res):
        pbar_norm_tasks.update()

    for i in range(totalH):
        pool.apply_async(normDnV, args=(i, totalH, folder, inD_min, inD_max, outD_min, outD_max), callback=pbar_norm_tasks_update)

    pool.close()
    pool.join()
    pbar_norm_tasks.close()

    # Unity FBGraph
    overall_request_matrix = sum(request_matrices)
    print('\n-> Overall request matrix calculated.')

    overall_FN_graph, overall_BN_graph = constructFBGraph(overall_request_matrix, num_grid_nodes=grid_info['gridNum'], mix=False)
    unified_FBN_graphs_path = os.path.join(folder, 'FBGraphs.dgl')
    dgl.save_graphs(unified_FBN_graphs_path, [overall_FN_graph, overall_BN_graph])
    print('-> Unified Semantic Neighborhood graphs saved to %s' % unified_FBN_graphs_path)

    overall_mixFBN_graph = constructFBGraph(overall_request_matrix, num_grid_nodes=grid_info['gridNum'], mix=True)
    unified_mixFBN_graph_path = os.path.join(folder, 'FBGraphMix.dgl')
    dgl.save_graphs(unified_mixFBN_graph_path, overall_mixFBN_graph)
    print('-> Unified Semantic Neighborhood graph (mixed for GEML) saved to %s' % unified_mixFBN_graph_path)

    print('Data splitting complete.')


if __name__ == '__main__':
    """
    Usage Example:
        python DataPackager.py -d ny2016_0101to0331.csv --minLat 40.4944 --maxLat 40.9196 --minLng -74.2655 --maxLng -73.6957 --refGridH 2.5 --refGridW 2.5 -er 1 -od ../data/ -c 10
    """
    # Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='ny2016_0101to0331.csv',
                        help='The input request data file to be handled, default={}'.format('ny2016_0101to0331.csv'))
    parser.add_argument('--minLat', type=float, default=40.4944,
                        help='The minimum latitude for the grids, default={}'.format(40.4944))
    parser.add_argument('--maxLat', type=float, default=40.9196,
                        help='The maximum latitude for the grids, default={}'.format(40.9196))
    parser.add_argument('--minLng', type=float, default=-74.2655,
                        help='The minimum longitude for the grids, default={}'.format(-74.2655))
    parser.add_argument('--maxLng', type=float, default=-73.6957,
                        help='The minimum latitude for the grids, default={}'.format(-73.6957))
    parser.add_argument('--refGridH', type=float, default=2.5,
                        help='The reference height for the grids, default={}, final grid height might be different'.format(
                            2.5))
    parser.add_argument('--refGridW', type=float, default=2.5,
                        help='The reference height for the grids, default={}, final grid width might be different'.format(
                            2.5))
    parser.add_argument('-er', '--exportRequests', type=int, default=1,
                        help='Whether the split requests should be exported, default={}'.format(1))
    parser.add_argument('-od', '--outDir', type=str, default='./',
                        help='Where the data should be exported, default={}'.format('""'))
    parser.add_argument('-c', '--cores', type=int, default=10,
                        help='How many cores should we use for paralleling , default={}'.format(10))
    FLAGS, unparsed = parser.parse_known_args()

    if not os.path.isfile(FLAGS.data):
        print('Data file path {} is invalid.'.format(FLAGS.data))
        exit(-1)

    if not os.path.isdir(FLAGS.outDir):
        print('Output path {} is invalid.'.format(FLAGS.outDir))
        exit(-2)

    folderName = path2FileNameWithoutExt(FLAGS.data)
    folderName = os.path.join(FLAGS.outDir, folderName)
    if not os.path.isdir(folderName):
        os.mkdir(folderName)

    # 1
    gridInfo = getGridInfo(FLAGS.minLat, FLAGS.maxLat, FLAGS.minLng, FLAGS.maxLng, FLAGS.refGridH, FLAGS.refGridW)
    saveGridInfo(gridInfo, os.path.join(folderName, 'grid_info.json'))
    # print(json.load(open(os.path.join(folderName, 'grid_info.json'))))    # Load Example

    # 2
    gridNodes = makeGridNodes(gridInfo)
    geoGraph = getGeoGraph(gridNodes, max(gridInfo['gridH'], gridInfo['gridW']) * 1.05)
    saveGeoGraph(geoGraph, os.path.join(folderName, 'GeoGraph.dgl'))
    # print(dgl.load_graphs(os.path.join(folderName, 'GeoGraph.dgl')))  # Load Example

    # 3
    splitData(FLAGS.data, folderName, gridNodes, gridInfo, FLAGS.exportRequests == 1, num_workers=FLAGS.cores)
    # print(np.load('GVQ.npy', allow_pickle=True).item())  # Load Example
    # (fg, bg), _ = dgl.load_graphs('FBGraphs.dgl')  # Load Example
