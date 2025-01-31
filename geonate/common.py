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
    """Return boundary location (left, bottom, right, top) of raster image for cropping image

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



