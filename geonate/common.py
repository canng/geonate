"""
The common module contains common functions and classes used by the other modules.

"""
# import common packages 

from typing import AnyStr, Dict, Optional

##############################################################################################
#                                                                                                                                                                                                          #
#                       Main functions                                                                                                                                                         #
#                                                                                                                                                                                                           #
##############################################################################################

# =========================================================================================== #
#               Create an empty dataframe                                                                                                                                           #
# =========================================================================================== #
def empty_dataframe(nrows, ncols, value='NA', name=None):
    """Create an empty dataframe

    Args:
        nrows (numeric): Numbers of rows
        ncols (numeric): Number of columns
        value (str | numeric, optional): Input value in all cells. Defaults to 'NA'.
        name (list, optional): Names of columns, if not given, it will return default as number of column. Defaults to None.

    Returns:
        pandas dataframe: An empty filled with NA or user-defined number (e.g., 0)

    """
    import pandas as pd
    import numpy as np
    
    # Check validity of column name
    if name is None:
        column_names = [f'Col_{i+1}' for i in range(ncols)]
    elif len(name) == ncols:
        column_names = name
    else:
        raise ValueError("Length of column names vector must match numbers of columns")

    # check input value
    try: 
        if isinstance(value, int):
            val = value
        elif isinstance(value, float):
            val = value
        else:
            val = np.nan
    except ValueError:
        val = np.nan
    
    # Create data and parse it into dataframe 
    data = [[val] * ncols for _ in range(nrows)]
    dataframe = pd.DataFrame(data, columns= column_names)
    
    return dataframe


# =========================================================================================== #
#               Find all files in folder with specific pattern                                                                                                                  #
# =========================================================================================== #
def listFiles(path: AnyStr, pattern: AnyStr, search_type: AnyStr = 'pattern', full_name: bool=True):
    """List all files with specific pattern within a folder path

    Args:
        path (AnyStr): Folder path where files stored
        pattern (AnyStr): Search pattern of files (e.g., '*.tif')
        search_type (AnyStr, optional): Search type whether by "extension" or name "pattern". Defaults to 'pattern'.
        full_name (bool, optional): Whether returning full name with path detail or only file name. Defaults to True.

    Returns:
        list: A list of file paths

    """
    import os
    import fnmatch

    # Create empty list to store list of files
    files_list = []

    # Check search type
    if (search_type.upper() == 'EXTENSION') or (search_type.upper() == 'E'):
        if '*' in pattern:
            raise ValueError("Do not use '*' in the pattern of extension search")
        else:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(pattern):
                        if full_name is True:
                            files_list.append(os.path.join(root, file))
                        else:
                            files_list.append(file)    
    
    elif (search_type.upper() == 'PATTERN') or (search_type.upper() == 'P'):
        if '*' not in pattern:
            raise ValueError("Pattern search requires '*' in pattern")
        else:
            for root, dirs, files in os.walk(path):
                for file in fnmatch.filter(files, pattern):
                    if full_name is True:
                        files_list.append(os.path.join(root, file))
                    else:
                        files_list.append(file)
    
    else:
        raise ValueError('Search pattern must be one of these types (pattern, p, extension, e)')

    return files_list


# =========================================================================================== #
#               Get general extent                                                                                                                                                           #
# =========================================================================================== #
def extent(input: AnyStr, poly: bool= True):
    """Get spatial extent of geotif image from a list or local variable

    Args:
        input (list): An input as a list of geotif files or local image/shapefile
        poly (bool, optional): Whether returns the extent polygon as geopandas object. Defaults to True.

    Returns:
        extent: Bounding box in form of BoundingBox(left, bottom, right, top)
        polygon: Geospatial shapefile polygon of the outside extent
    
    """
    import rasterio
    import geopandas as gpd
    from shapely.geometry import Polygon

    general_extent = None

    # get extent for raster files store in folder
    if (isinstance(input, list)) or (isinstance(input, str)):
        for file in input:
            with rasterio.open(file) as src:
                ext = src.bounds
                crs = src.crs

                if general_extent is None:
                    general_extent = ext
                else:
                    general_extent =  (
                        min(general_extent[0], ext[0]),
                        min(general_extent[1], ext[1]),
                        max(general_extent[2], ext[2]),
                        max(general_extent[3], ext[3])
                        )
                    
    # get extent for local read shapefile    
    elif isinstance(input, gpd.GeoDataFrame):
        ext = input.bounds
        crs = input.crs
        general_extent = (ext['minx'], ext['miny'],
                                        ext['maxx'], ext['maxy'])
    
    # get extent for local read geotif 
    else:
        general_extent = input.bounds
        crs = input.crs
    
    # return rectangle of extennt
    if poly is True:
        poly_geom = Polygon([
            (general_extent[0], general_extent[1]), 
            (general_extent[2], general_extent[1]), 
            (general_extent[2], general_extent[3]), 
            (general_extent[0], general_extent[3])
            ])
        poly = gpd.GeoDataFrame(index=[0], geometry=[poly_geom])
        poly.crs = {'init': crs}
    else: 
        poly = None

    return general_extent, poly
    

# =========================================================================================== #
#               Get bounds of raster
# =========================================================================================== #
def getBounds(input: AnyStr, meta: Optional[Dict]=None):
    """Return boundary location (left, bottom, right, top) of raster image for cropping image in number list

    Args:
        input (AnyStr): Image or data array input 
        meta (Dict, optional): Metadata is needed when input is data array. Defaults to None.

    Returns:
        numeric: A list of number show locations of left, bottom, right, top of the boundary

    Example:
        img = raster.rast('./Sample_data/landsat_multi/Scene/landsat_img_00.tif')
        meta = img.meta
        ds = img.read()
        left, bottom, right, top = raster.getBounds(ds, meta)

    """
    import rasterio
    import numpy as np

    # Check input 
    if isinstance(input, rasterio.DatasetReader):
        left, bottom, right, top = input.bounds
    # input is array
    elif isinstance(input, np.ndarray):
        if meta is None:
            raise ValueError('It requires metadata of input')
        else:
            transform = meta['transform']
            width = meta['width']
            height = meta['height']
            left, top = transform * (0, 0)
            right, bottom = transform * (width, height)

    # Other input
    else:
        raise ValueError('Input data is not supported')
    
    return left, bottom, right, top


# =========================================================================================== #
#              Convert meter to acr-degree based on latitude
# =========================================================================================== #
def meter2degree(input, latitude=None):
    """Convert image resolution from meter to acr-degree depending on location of latitude

    Args:
        input (numeric): Input resolution of distance
        latitude (numeric, optional): Latitude presents location. If latitude is None, the location is assumed near Equator. Defaults to None.

    Returns:
        numeric: Degree corresponding to the distance length

    """
    import numpy as np

    if latitude is None:
        # Equator location
        degree = input / (111320 * np.cos(np.radians(0.0)))
    else:
        degree = input / (111320 * np.cos(np.radians(latitude)))
    
    return degree


# =========================================================================================== #
#              Return min and max values of array or raster
# =========================================================================================== #
def mimax(input, digit=3):
    """Calculate maximum and minimum values of raster or array

    Args:
        input (DatasetReader | np.ndarray): Rasterio image or data array
        digit (int, optional): Precise digit number. Defaults to 3.

    Returns:
        numeric: Return 2 numbers of minvalue and maxvalue

    """
    import rasterio
    import numpy as np
    import pandas as pd

    ### Check input data
    if isinstance(input, rasterio.DatasetReader):
        dataset = input.read()
    elif isinstance(input, np.ndarray):
        dataset = input
    else:
        raise ValueError('Input data is not supported')
    
    # Calculate min and max values
    minValue = round(np.nanmin(dataset), digit)
    maxValue = round(np.nanmax(dataset), digit)

    # Convert min and max to string for print
    min_round = str(round(minValue, digit))
    max_round = str(round(maxValue, digit))

    print(f"Min: {min_round} \nMax: {max_round}")

    return minValue, maxValue


# =========================================================================================== #
#              Estimate raster cell area 
# =========================================================================================== #
def cellSize(input, unit: AnyStr='km', meta: Optional[AnyStr]=None, output: Optional[AnyStr]=None):
    """Calculate pixel size (area), the input has to be in the projection of 'EPSG:4326'. If not, it can be reprojected by "project" function

    Args:
        input (DatasetReader | np.ndarray): Rasterio image or data array
        unit (AnyStr, optional): Unit of original input. Defaults to 'km'.
        meta (AnyStr, optional): Metadata in case input is data array. Defaults to None.
        output (AnyStr, optional): File path to write out geotif file to local directory. Defaults to None.

    Returns:
        np.darray: Data array contains all image pixel values
        metadata: Metedata of the image

    """
    import rasterio
    import numpy as np
    import pandas as pd
    import geonate import raster

    ### Check input data
    if isinstance(input, rasterio.DatasetReader):
        if len(input.shape) == 2:
            dataset = input.read()
            meta = input.meta
        else:
            dataset = input.read(1)
            meta = input.meta
    elif isinstance(input, np.ndarray):
        if meta is None:
            raise ValueError('Please provide input metadata')
        else:
            if len(input) > 2:
                dataset = input[0, :, : ]
            else:
                dataset = input
            meta = meta
    else:
        raise ValueError('Input data is not supported')
    
    ### Read metadata
    transform = meta['transform']
    pix_width = transform[0]
    upper_X = transform[2]
    upper_Y = transform[5]
    rows = meta['height']
    cols = meta['width']
    lower_X = upper_X + transform[0] * cols
    lower_Y = upper_Y + transform[4] * rows

    lats = np.linspace(upper_Y, lower_Y, rows + 1)

    a = 6378137.0  # Equatorial radius
    b = 6356752.3142  # Polar radius

    # Degrees to radians
    lats = lats * np.pi/180

    # Intermediate vars
    e = np.sqrt(1-(b/a)**2)
    sinlats = np.sin(lats)
    zm = 1 - e * sinlats
    zp = 1 + e * sinlats

    # Distance between meridians
    q = pix_width/360

    # Compute areas for each latitude in square km
    areas_to_equator = np.pi * b**2 * ((2*np.arctanh(e*sinlats) / (2*e) + sinlats / (zp*zm))) / 10**6
    areas_between_lats = np.diff(areas_to_equator)
    areas_cells = np.abs(areas_between_lats) * q

    # Create empty array to store output
    cellArea = np.zeros_like(dataset, dtype=np.float32)
    
    # Assign estimated cell area to every pixel
    if len(cellArea.shape) == 2:
        for i in range(0, cellArea.shape[1]):
            cellArea[:, i] = areas_cells.flatten()
    else:
        for i in range(0, cellArea.shape[2]):
            cellArea[:, :, i] = areas_cells.flatten()

    ### Update metadata
    meta.update({'dtype': np.float32, 'count': 1})

    ### Convert unit (if applicable)
    if (unit.lower() == 'km') or (unit.lower() == 'kilometer'):
        outArea = cellArea
    elif (unit.lower() == 'm') or (unit.lower() == 'meter'):
        outArea = cellArea * 1_000_000
    elif (unit.lower() == 'ha') or (unit.lower() == 'hectare'):
        outArea = cellArea * 10_000
    
    # Write output
    if output is not None:
        raster.writeRaster(outArea, output, meta)
    else:
        return outArea, meta

import numpy as np
import rasterio
from typing import AnyStr, Dict, Optional
from rasterio.io import MemoryFile


# =========================================================================================== #
#              Convert a numpy array and metadata to a rasterio object stored in local variable
# =========================================================================================== #
def array2raster(array: np.ndarray, meta: Dict) -> rasterio.io.DatasetReader:
    """
    Convert a numpy array and metadata to a rasterio object stored in memory.

    Args:
        array (np.ndarray): The input data array.
        meta (Dict): The metadata dictionary.

    Returns:
        rasterio.io.DatasetReader: The rasterio object stored in memory.

    """
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile

    # Update metadata with the correct dtype and count
    meta.update({
        'dtype': array.dtype,
        'count': array.shape[0] if array.ndim == 3 else 1
    })
    
    with MemoryFile() as memfile:
        with memfile.open(**meta) as dataset:
            dataset.write(array, 1 if array.ndim == 2 else None)
        return memfile.open()