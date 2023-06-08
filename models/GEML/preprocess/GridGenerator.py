"""
Generate grids with a given rectangle region as input
Output Format in csv: (gridID, midLat, midLng, gridLat, gridLng, gridH, gridW)
"""
import math
import pandas as pd


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


class GridGenerator:
    def __init__(self, minLat, maxLat, minLng, maxLng, refGridW=2.5, refGridH=2.5):
        """
        :param minLat: lower boundary of region's latitude
        :param maxLat: upper boundary of region's latitude
        :param minLng: lower boundary of region's longitude
        :param maxLng: upper boundary of region's longitude
        :param refGridW: reference width of a grid in km, will auto-adjust later
        :param refGridH: reference height of a grid in km, will auto-adjust later
        """
        self.minLat = minLat
        self.maxLat = maxLat
        self.minLng = minLng
        self.maxLng = maxLng
        self.gridH = refGridH
        self.gridW = refGridW

        print('Initial States:', self.__dict__)
        self.latLen = haversine((self.minLat, self.maxLng), (self.maxLat, self.maxLng))
        self.latGridNum = round(self.latLen / self.gridH)
        self.gridH = self.latLen / self.latGridNum
        self.gridLat = (self.maxLat - self.minLat) / self.latGridNum

        self.lngLen = haversine((self.maxLat, self.minLng), (self.maxLat, self.maxLng))
        self.lngGridNum = round(self.lngLen / self.gridW)
        self.gridW = self.lngLen / self.lngGridNum
        self.gridLng = (self.maxLng - self.minLng) / self.lngGridNum

        self.gridNum = self.latGridNum * self.lngGridNum

        self.grids = pd.DataFrame(columns=['gridID', 'midLat', 'midLng', 'gridLat', 'gridLng', 'gridH', 'gridW'])
        self.makeGrid()
        print('Processed States:', self.__dict__)

    def makeGrid(self):
        leftLng = self.minLng + self.gridLng / 2
        midLat = self.maxLat - self.gridLat / 2
        gridID = 0
        for i in range(self.latGridNum):
            midLng = leftLng
            for j in range(self.lngGridNum):
                self.grids = self.grids.append({
                    'gridID': gridID,
                    'midLat': midLat,
                    'midLng': midLng,
                    'gridLat': self.gridLat,
                    'gridLng': self.gridLng,
                    'gridH': self.gridW,
                    'gridW': self.gridH,
                }, ignore_index=True)
                gridID += 1
                midLng += self.gridLng
            midLat -= self.gridLat

    def save(self, fName):
        self.grids.to_csv(fName, index=False)
        print('Grids saved to {}'.format(fName))

    def saveParam(self, fName):
        f = open(fName, 'w')
        f.write(self.__dict__.__str__())
        f.close()
        print('Parameters saved to {}'.format(fName))


# Test
if __name__ == '__main__':
    nyGridGen = GridGenerator(40.4944, 40.9196, -74.2655, -73.6957, 2.5, 2.5)
    nyGridGen.save('nyGrids.csv')
    nyGridGen.saveParam('nyGrids.param')
