"""
Please do not remove these lines
Authored by George Azzari
A few small changes by Chris Yeh and Anthony Perez
"""

import gdal, osr
import numpy as np
import pandas as pd


class GeoProps(object):
    def __init__(self):
        self.eDT = None
        self.Proj = None
        self.GeoTransf = None
        self.Driver = None
        self.Flag = False
        self.xOrigin = None
        self.yOrigin = None
        self.pixelWidth = None
        self.pixelHeight = None
        self.srs = None
        self.srsLatLon = None

    def import_geogdal(self, gdal_dataset):
        """
        adfGeoTransform[0] /* top left x */
        adfGeoTransform[1] /* w-e pixel resolution */
        adfGeoTransform[2] /* 0 */
        adfGeoTransform[3] /* top left y */
        adfGeoTransform[4] /* 0 */
        adfGeoTransform[5] /* n-s pixel resolution (negative value) */
        :param gdal_dataset: a gdal dataset
        :return: nothing, set geooproperties from input dataset.
        """
        gdal.SetCacheMax(2**30)
        self.eDT = gdal_dataset.GetRasterBand(1).DataType
        self.Proj = gdal_dataset.GetProjection()
        self.GeoTransf = gdal_dataset.GetGeoTransform()
        self.Driver = gdal.GetDriverByName("GTiff")
        self.xOrigin = self.GeoTransf[0]
        self.yOrigin = self.GeoTransf[3]
        self.pixelWidth = self.GeoTransf[1]
        self.pixelHeight = self.GeoTransf[5]
        self.srs = osr.SpatialReference()
        self.srs.ImportFromWkt(self.Proj)
        self.srsLatLon = self.srs.CloneGeogCS()
        self.Flag = True

    def get_affinecoord(self, geolon, geolat):
        """Returns coordinates in meters (affine) from degrees coordinates (georeferenced)"""
        ct = osr.CoordinateTransformation(self.srsLatLon, self.srs)
        tr = ct.TransformPoint(geolon, geolat)
        xlin = tr[0]
        ylin = tr[1]
        return xlin, ylin

    def get_georefcoord(self, xlin, ylin):
        """Returns coordinates in degrees (georeferenced) from coordinates in meters (affine)"""
        ct = osr.CoordinateTransformation(self.srs, self.srsLatLon)
        tr = ct.TransformPoint(xlin, ylin)
        geolon = tr[0]
        geolat = tr[1]
        return geolon, geolat

    def lonlat2colrow(self, lon, lat):
        """Returns the [col, row] of a pixel given its lon, lat coordinates
        NOTE: the lon/lat coordinates must have the same units (e.g. degrees or meters) as
        self.xOrigin, self.pixelWidth, etc.
        """
        col = int((lon - self.xOrigin) / self.pixelWidth)
        row = int((lat - self.yOrigin) / self.pixelHeight)
        # print "(long,lat) = (",GeoX, ",", GeoY,") --> (col,row) = (",xOffset,",",yOffset,")"
        # NOTE: watch out! if you're using this to read a 2D np.array, remember
        # that xOffset = col, yOffset = row
        return [col, row]

    def lonlat2colrow_batch(self, lon, lat):
        """Vectorized version of lonlat2colrow
        Args:
            lon: 1-D np.array of longitudes
            lat: 1-D np.array of latitudes (same length as lon)
        NOTE: the lon/lat coordinates must have the same units (e.g. degrees or meters) as
        self.xOrigin, self.pixelWidth, etc.
        Returns: tuple of (col, row) which are 1-D np.arrays of ints
        """
        col = np.int_((lon - self.xOrigin) / self.pixelWidth)
        row = np.int_((lat - self.yOrigin) / self.pixelHeight)
        # Bug fix, if lon or lat is a 1-D array of shape (1,) then a bug will occur where the output is
        # an integer instead of an array
        if lon.shape[0] == 1:
            col = np.array([col])
        if lat.shape[0] == 1:
            row = np.array([row])
        return col, row

    def colrow2lonlat(self, col, row):
        """ Returns the (lon, lat) of a pixel given its (col, row)
        NOTE: the returned lon/lat coordinates will have the same units (e.g. degrees or
        meters) as self.xOrigin, self.pixelWidth, etc.
        """
        lon = col * self.pixelWidth + self.xOrigin
        lat = row * self.pixelHeight + self.yOrigin
        return [lon, lat]

    def get_center_coord(self, raster_array_shape, affine=False):
        """ Input: raster_array_shape is the output of gdalobject.np_array.shape, which is (#rows, #cols)
            Returns: coordinate (lon, lat) of the center of the raster."""
        s = raster_array_shape
        ul = self.colrow2lonlat(0, 0)
        lr = self.colrow2lonlat(s[1], s[0])
        lon_ext_m = lr[0] - ul[0]
        lat_ext_m = ul[1] - lr[1]
        lon_cntr = ul[0] + lon_ext_m / 2
        lat_cntr = lr[1] + lat_ext_m / 2
        if affine:
            return lon_cntr, lat_cntr
        if not affine:
            return self.get_georefcoord(lon_cntr, lat_cntr)

    def get_raster_extent(self, raster_array_shape):
        """ Input: raster_array_shape is the output of gdalobject.np_array.shape, which is (#rows, #cols)
            Returns: extent (in meters) of the raster."""
        s = raster_array_shape
        ul = self.colrow2lonlat(0, 0)
        lr = self.colrow2lonlat(s[1], s[0])
        lon_ext_m = lr[0] - ul[0]
        lat_ext_m = ul[1] - lr[1]
        return lon_ext_m, lat_ext_m

    def get_small_pxlwin(self, lon, lat, dpx, dpy):
        cen_col, cen_row = self.lonlat2colrow(lon, lat)
        rows = range(cen_row - dpy, cen_row + dpy + 1, 1)
        columns = range(cen_col - dpx, cen_col + dpx + 1, 1)
        row_indx = []
        col_indx = []
        for i in rows:
            row_indx.append(i)
        for j in columns:
            col_indx.append(j)
        return np.array(row_indx), np.array(col_indx)


class ImageComposite(object):

    def __init__(self, imgpath, nbands):

        self.imgpath = imgpath
        self.gdal_dataset = gdal.Open(imgpath)
        self.geoprops = GeoProps()
        self.geoprops.import_geogdal(self.gdal_dataset)
        self.bands = range(1, nbands)
        self.nbands = nbands

    def getpxwin(self, lon, lat, nrows, ncols, fpath, addgeo=True):

        rowindx, colindx = self.geoprops.get_small_pxlwin(lon, lat, ncols/2, nrows/2)
        # print("Max rowindx: ", rowindx.max())
        # print("Min rowindx: ", rowindx.min())
        # print("Max colindx: ", colindx.max())
        # print("Min colindx: ", colindx.min())
        ul = self.geoprops.colrow2lonlat(colindx.min(), rowindx.min())
        # print('GeoT: ', [ul[0], self.geoprops.pixelWidth, 0, ul[1], 0, self.geoprops.pixelHeight])
        # print('ul coords: ', ul)

        # One of Byte, UInt16, Int16, UInt32, Int32, Float32, Float64,
        # and the complex types CInt16, CInt32, CFloat32, and CFloat64.
        gdal_datatype = self.geoprops.eDT
        driver = self.geoprops.Driver
        dst_ds = driver.Create(fpath, ncols, nrows, self.nbands, gdal_datatype)

        if addgeo:
            dst_ds.SetGeoTransform([ul[0], self.geoprops.pixelWidth, 0, ul[1], 0, self.geoprops.pixelHeight])
            dst_ds.SetProjection(self.geoprops.Proj)

        # print(ul[0], ul[1], ncols, nrows)
        for i in self.bands:
            bandarray = self.gdal_dataset.GetRasterBand(i).ReadAsArray(colindx.min(), rowindx.min(), ncols, nrows)
            # print(np.all((bandarray == 0) + (bandarray == -32768)))
            # bandarray = bandarray.astype(np.float32)
            # print(np.all((bandarray == 0) + (bandarray == -32768)))
            dst_ds.GetRasterBand(i).WriteArray(bandarray)

        dst_ds = None

    def _pdrowfu(self, pdrow, lonindx, latindx, nrows, ncols, prefix, suffix, basepath, verbose=True):

        lon = pdrow[lonindx]
        lat = pdrow[latindx]

        if verbose:
            print('Tiling {0}x{1} image around (lat, lon)=({2}, {3})'.format(str(nrows), str(ncols),
                                                                             str(round(lat, 2)), str(round(lon, 2))))

        rname = 'ROW{0}_LON{1}_LAT{2}'.format(str(int(pdrow[0])), str(round(lon, 2)), str(round(lat, 2)))
        fname = "{0}_{1}_{2}.tif".format(prefix, rname, suffix)

        self.getpxwin(pdrow[lonindx], pdrow[latindx], nrows, ncols, "{0}/{1}".format(basepath, fname))

    def getgridwins(self, gridpath, lonindx, latindx, nrows, ncols, prefix, suffix, exportpath):

        p = pd.read_csv(gridpath)
        p.apply(self._pdrowfu, axis=1, raw=True, args=(lonindx, latindx, nrows, ncols, prefix, suffix, exportpath))
