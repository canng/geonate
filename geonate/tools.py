from typing import AnyStr, Dict, Optional
from shapely.geometry import polygon
from shapely import speedups
speedups.disable()

import os
os.environ['GDAL_DATA'] = os.environ['CONDA_PREFIX'] + r'\Library\share\gdal'
os.environ['PROJ_LIB'] = os.environ['CONDA_PREFIX'] + r'\Library\share'


##############################################################################################
'''
TOOL FUNCTIONS

1. list_files
2. empty_dataframe
2. getExtent


'''
# =========================================================================================== #
#               Create an empty dataframe
# =========================================================================================== #
def empty_dataframe(nrows, ncols, names=None):
    '''
    Create an empty dataframe

    Parameters:
        nrows: numeric, numbers of rowns
        ncols: numeric, number of columns
        names: vector, names of columns, if not given, it will return default as number of column
        
    Example:
        data = tools.empty_dataframe(nrows= 5, ncols= 2, names= ['Col1', 'Col2'])
    '''
    import pandas as pd
    import numpy as np
    
    if names is None:
        column_names = [f'Column{i+1}' for i in range(ncols)]
    elif len(names) == ncols:
        column_names = names
    else:
        raise ValueError("Length of column names vector must match numbers of columns")

    data = [[np.nan] * ncols for _ in range(nrows)]
    dataframe = pd.DataFrame(data, columns=column_names)
    
    return dataframe


# =========================================================================================== #
#               Find all files in folder with specific pattern
# =========================================================================================== #
def list_files(path: AnyStr, pattern: AnyStr, search_type: AnyStr = 'pattern', full_name: bool=True):
    '''
    List all files with specific pattern within a folder path

    Parameters:
        path: string, folder path where files storing
        pattern: string, search pattern of files (e.g., '*.tif')
        search_type: string, search type whether by extension or name pattern
        full_name: boolean, whether returning full name
        
    Example:
        files = tools.list_files(path= './Sample_data/shapefile/', pattern='*shp', search_type='e', full_name= False)

    '''

    import os
    import fnmatch

    files_list = []
    if (search_type.lower() == 'extension') or (search_type.lower() == 'e'):
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
    elif (search_type.lower() == 'pattern') or (search_type.lower() == 'p'):
        if '*' not in pattern:
            raise ValueError("Pattern search requires '*' in the pattern")
        else:
            for root, dirs, files in os.walk(path):
                for file in fnmatch.filter(files, pattern):
                    if full_name is True:
                        files_list.append(os.path.join(root, file))
                    else:
                        files_list.append(file)
    else:
        raise ValueError('search pattern are: pattern, p, extention, e')

    return files_list

# =========================================================================================== #
#               Get general extent
# =========================================================================================== #
def extent(input: AnyStr, poly: bool= True):
    '''
    Return the extent of geotif image from the list or local variable

    Parameters:
        input: string, an input as a list of geotif files or local image/shapeile
        poly: bool, whether returns the extent polygon as geopandas object
        
    Example:
        files = tools.list_files('./landsat_multi/merge/', 'tif')
        ext, poly = tools.extent(files)

    '''
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
#              Convert meter to acr-degree based on latitude
# =========================================================================================== #
def meter2degree(input, latitude):
    '''
    Convert image resolution from meter to acr-degree depending on location of latitude

    Parameters:
        input: number, input resolution of distance
        latitude: number, latitude presents location 
    
    Example:
       degree = raster.meter2degree(30, 10.25)   

    '''
    import numpy as np
    degree = input / (111320 * np.cos(np.radians(latitude)))
    
    return degree



    



