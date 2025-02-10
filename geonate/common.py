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
        Dataframe (pandas datafram): An empty filled with NA or user-defined number (e.g., 0)

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
        A string list (list): A list of file paths

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
#               Checks if all elements in the input list have the same file extension.                                                                                                                  
# =========================================================================================== #
def check_extension_consistency(input):
    """
    Checks if all elements in the input list have the same file extension.

    Args:
        input (list): A list of file paths as strings.

    Returns:
        tuple: A tuple containing:
            - bool: True if all elements have the same file extension, False otherwise.
            - str or None: The file extension if all elements have the same extension, otherwise None.

    """
    import numpy

    # Check whether a list or not
    if not isinstance(input, list):
        raise ValueError("Input must be a list of file paths")
    
    else:
        # Check whether all are string
        if not all(isinstance(x, str) for x in input):
            raise ValueError('Input must contain only string of file paths')
        else:
            # 
            extensions = [e.split(".")[-1] for e in input]
            no_extension = len(numpy.unique(extensions))
            if no_extension == 1:
                consistency = True
                extension = str(extensions[0])
            else:
                consistency = False
                extension = None
            
            return consistency, extension          

# =========================================================================================== #
#               Checks if all elements in the input list have the same data type.                                                                                                                  
# =========================================================================================== #
def check_datatype_consistency(input):
    """
    Checks if all elements in the input list have the same data type.

    Args:
        input (list): A list of elements to check.

    Returns:
        tuple: A tuple containing:
            - bool: True if all elements have the same type, False otherwise.
            - type or None: The data type of the elements if all elements have the same type, otherwise None.

    """
    if not isinstance(input, list):
        raise ValueError("Input must be a list")
    else:
        first_element = type(input[0])
        if all(isinstance(item, first_element) for item in input):
            datatype = first_element
            consistency = True
            print(f"Checking datatype consistency\nInput have data types of {datatype}")
        else:
            datatype = None
            consistency = False
            print(f"Checking datatype consistency\nInput have different data types")
        
        return consistency, str(datatype)


# =========================================================================================== #
#               Checks if all elements in the input list have the same Coordinate Reference System (CRS).                                                                                                           
# =========================================================================================== #
def check_crs_consistency(input):
    """
    Checks if all elements in the input list have the same Coordinate Reference System (CRS).

    Args:
        input (list): A list of file paths or local variables (rasterio.io.DatasetReader or geopandas.geodataframe.GeoDataFrame).

    Returns:
        tuple: A tuple containing:
            - bool: True if all elements have the same CRS, False otherwise.
            - str or None: The CRS of the elements if all elements have the same CRS, otherwise None.
    
    """
    import numpy
    from .geonate import rast, vect
    
    if not isinstance(input, list):
        raise ValueError('Input must be a list of local variables or file paths')
    
    else:
        _, inputType = check_datatype_consistency(input)
        
        if str(inputType) == "<class 'str'>":
            file_extensions  = [x.split(".")[-1] for x in input]
            file_extension = numpy.unique(file_extensions)
            extension_len = len(file_extension)

            if extension_len > 1:
                raise ValueError('Input must have consistent data format')
            
            else:
                if str(file_extension[0]) == 'tif':
                    files = [rast(file) for file in input]
                    crs_list = [x.crs.to_string() for x in files]

                    if len(numpy.unique(crs_list)) == 1:
                        crs = numpy.unique(crs_list)
                        consistency = True
                        print(f"Input is Raster with consistent crs of {crs}")
                        return consistency, crs[0]
                    
                    else:
                        consistency = False
                        print(f"Input is Raster with different crs")
                        return consistency, None

                elif str(file_extension[0]) == 'shp':
                    files = [vect(file) for file in input]
                    crs_list = [file.crs.to_string() for file in files]
                    if len(numpy.unique(crs_list)) == 1:
                        crs = numpy.unique(crs_list)
                        consistency = True
                        print(f"Input is Shapefile with consistent crs of {crs}")
                        return consistency, crs[0]
                    
                    else:
                        consistency = False
                        print(f"Input is Raster with different crs")
                        return consistency, None

                else:
                    raise ValueError('This function only supports raster (.tif) and shapefile (.shp)')

        elif str(inputType) == "<class 'rasterio.io.DatasetReader'>" or str(inputType) == "<class 'geopandas.geodataframe.GeoDataFrame'>":
            crs_list = [x.crs.to_string() for x in input]
            crs = numpy.unique(crs_list)
            consistency = True
            print(f"Input is Shapefile with consistent crs of {crs}")
            return consistency, crs[0]

        else:
            raise ValueError(f"Input must have the same data format of raster (.tif) and shapefile (.shp)")
            

# =========================================================================================== #
#               Computesspatial extent of a single or multiple geospatial files (local already read variables)                                                                                                      
# =========================================================================================== #   
def get_extent_local(input):
    """
    Computes the spatial extent of a single or multiple geospatial files (GeoTIFF raster or shapefile).

    Args:
        input (list or object): A single rasterio.io.DatasetReader object, a single geopandas.geodataframe.GeoDataFrame object, or a list of such objects.

    Returns:
        tuple: A tuple containing:
            - general_extent (tuple): The bounding box of the input(s) in the format (min_x, min_y, max_x, max_y).
            - bound_poly (geopandas.GeoDataFrame): A GeoDataFrame containing a single polygon representing the bounding box.

    """
    import rasterio
    import geopandas
    from shapely.geometry import Polygon

    #### Single file
    if (not isinstance(input, list)) or len(input)==1:
        if (isinstance(input, rasterio.io.DatasetReader)):
            general_extent = input.bounds
            crs = input.crs

            poly_geom = Polygon([
                    (general_extent[0], general_extent[1]), 
                    (general_extent[2], general_extent[1]), 
                    (general_extent[2], general_extent[3]), 
                    (general_extent[0], general_extent[3])
                    ])
            bound_poly = geopandas.GeoDataFrame(index=[0], geometry=[poly_geom])
            bound_poly.crs = {'init': crs}

            return general_extent, bound_poly

        elif (isinstance(input, geopandas.geodataframe.GeoDataFrame)):
            ext = input.bounds
            crs = input.crs
            no_poly = ext.shape[0]
                
            # Determine general bound in case single polygon or multiple polygons
            if no_poly == 1:
                general_extent = tuple(ext.loc[0, :])
            elif no_poly >= 2:
                general_extent = (ext.iloc[:, 0].min(), ext.iloc[:, 1].min(), ext.iloc[:, 2].max(),ext.iloc[:, 3].max())

            poly_geom = Polygon([
                    (general_extent[0], general_extent[1]), 
                    (general_extent[2], general_extent[1]), 
                    (general_extent[2], general_extent[3]), 
                    (general_extent[0], general_extent[3])
                    ])
            bound_poly = geopandas.GeoDataFrame(index=[0], geometry=[poly_geom])
            bound_poly.crs = {'init': crs}

            return general_extent, bound_poly
            
        else:
            raise ValueError('It only supports geotif raster and shapefile')
            
    #### Multiple files
    elif (isinstance(input, list)) or len(input) > 1:
        consistency, datatype = check_datatype_consistency(input)
        
        # Input are Raster files
        if (consistency is True) and (datatype == "<class 'rasterio.io.DatasetReader'>"):
            general_extent = None        
            # read each file
            for file in input:
                ext = file.bounds
                crs = file.crs                
                
                # determine general extent
                if general_extent is None:
                    general_extent = ext
                else:
                    general_extent =  (
                        min(general_extent[0], ext[0]),
                        min(general_extent[1], ext[1]),
                        max(general_extent[2], ext[2]),
                        max(general_extent[3], ext[3])
                        )
                    
                # Create bound polygon
                poly_geom = Polygon([
                    (general_extent[0], general_extent[1]), 
                    (general_extent[2], general_extent[1]), 
                    (general_extent[2], general_extent[3]), 
                    (general_extent[0], general_extent[3])
                    ])
                bound_poly = geopandas.GeoDataFrame(index=[0], geometry=[poly_geom])
                bound_poly.crs = {'init': crs} 

                return general_extent, bound_poly

        # Input are Shapefile files
        elif (consistency is True) and (datatype == "<class 'geopandas.geodataframe.GeoDataFrame'>"):
            general_extent = None                        
            # read each file
            for file in input:
                ext = file.bounds
                crs = file.crs
                no_poly = ext.shape[0]
    
                # Determine general bound in case single polygon or multiple polygons
                if no_poly == 1:
                    ext = tuple(ext.loc[0, :])
                elif no_poly >= 2:
                    ext = (ext.iloc[:, 0].min(), ext.iloc[:, 1].min(), ext.iloc[:, 2].max(),ext.iloc[:, 3].max())
                
                # determine general extent
                if general_extent is None:
                    general_extent = ext
                else:
                    general_extent =  (
                        min(general_extent[0], ext[0]),
                        min(general_extent[1], ext[1]),
                        max(general_extent[2], ext[2]),
                        max(general_extent[3], ext[3])
                        )
                
                # Create bound polygon
                poly_geom = Polygon([
                    (general_extent[0], general_extent[1]), 
                    (general_extent[2], general_extent[1]), 
                    (general_extent[2], general_extent[3]), 
                    (general_extent[0], general_extent[3])
                    ])
                bound_poly = geopandas.GeoDataFrame(index=[0], geometry=[poly_geom])
                bound_poly.crs = {'init': crs} 

                return general_extent, bound_poly
        
        # Other data types
        else:
            raise ValueError('Input have different data types') 
    
    #### Other cases
    else:
        raise ValueError('It only supports geotif raster and shapefile')


# =========================================================================================== #
#               Computes spatial extent of geospatial files (external files)   
# =========================================================================================== #   

def get_extent_external(input):
    """
    Computes the spatial extent of geospatial files and returns the bounding box and a GeoDataFrame of the bounding polygon.

    Args:
        input (str or list): A single file path or a list of file paths. Supported file types are GeoTIFF raster (tif) and shapefile (shp).

    Returns:
        tuple: A tuple containing:
            - general_extent (tuple): The bounding box of the input files in the format (minx, miny, maxx, maxy).
            - bound_poly (geopandas.GeoDataFrame): A GeoDataFrame containing the bounding polygon.

    """
    
    import rasterio
    import geopandas
    from shapely.geometry import Polygon
    from .geonate import rast, vect

    #### Single path
    if (not isinstance(input, list)) or ((isinstance(input, list) and (len(input)==1))):
        # Check whether single list or string
        if len(input) == 1:
            input = input[0]
        else:
            input = input

        # Extract file extension
        extension = input.split(".")[-1]

        # Raster files
        if extension == 'tif':
            tmp = rast(input)
            general_extent = tmp.bounds
            crs = tmp.crs
            # Create bound polygon
            poly_geom = Polygon([
                    (general_extent[0], general_extent[1]), 
                    (general_extent[2], general_extent[1]), 
                    (general_extent[2], general_extent[3]), 
                    (general_extent[0], general_extent[3])
                    ])
            bound_poly = geopandas.GeoDataFrame(index=[0], geometry=[poly_geom])
            bound_poly.crs = {'init': crs}

            return general_extent, bound_poly
        
        # Vector Shapefile 
        elif extension == 'shp':
            tmp = vect(input)
            ext = tmp.bounds
            crs = tmp.crs
            no_poly = ext.shape[0]
                
            # Determine general bound in case single polygon or multiple polygons
            if no_poly == 1:
                general_extent = tuple(ext.loc[0, :])
            elif no_poly >= 2:
                general_extent = (ext.iloc[:, 0].min(), ext.iloc[:, 1].min(), ext.iloc[:, 2].max(),ext.iloc[:, 3].max())

            # Create bound polygon
            poly_geom = Polygon([
                    (general_extent[0], general_extent[1]), 
                    (general_extent[2], general_extent[1]), 
                    (general_extent[2], general_extent[3]), 
                    (general_extent[0], general_extent[3])
                    ])
            bound_poly = geopandas.GeoDataFrame(index=[0], geometry=[poly_geom])
            bound_poly.crs = {'init': crs}

            return general_extent, bound_poly

        # Other data types
        else:
            raise ValueError('It only supports geotif raster (tif) and shapefile (shp)')

    #### List of multiple paths    
    else:
        consistency, extension = check_extension_consistency(input)
        
        # Check extension consistency
        if consistency is True:
            # Raster files
            if extension == 'tif':
                files = [rast(file) for file in input]
                general_extent = None 

                # read each file
                for file in files:
                    ext = file.bounds
                    crs = file.crs                
                    
                    # determine general extent
                    if general_extent is None:
                        general_extent = ext
                    else:
                        general_extent =  (
                            min(general_extent[0], ext[0]),
                            min(general_extent[1], ext[1]),
                            max(general_extent[2], ext[2]),
                            max(general_extent[3], ext[3])
                            )
                        
                    # Create bound polygon
                    poly_geom = Polygon([
                        (general_extent[0], general_extent[1]), 
                        (general_extent[2], general_extent[1]), 
                        (general_extent[2], general_extent[3]), 
                        (general_extent[0], general_extent[3])
                        ])
                    bound_poly = geopandas.GeoDataFrame(index=[0], geometry=[poly_geom])
                    bound_poly.crs = {'init': crs} 

                    return general_extent, bound_poly

            # Shapefile data
            elif extension == 'shp':
                files = [vect(file) for file in input]
                general_extent = None                        
            
                # read each file
                for file in files:
                    ext = file.bounds
                    crs = file.crs
                    no_poly = ext.shape[0]
        
                    # Determine general bound in case single polygon or multiple polygons
                    if no_poly == 1:
                        ext = tuple(ext.loc[0, :])
                    elif no_poly >= 2:
                        ext = (ext.iloc[:, 0].min(), ext.iloc[:, 1].min(), ext.iloc[:, 2].max(),ext.iloc[:, 3].max())
                    
                    # determine general extent
                    if general_extent is None:
                        general_extent = ext
                    else:
                        general_extent =  (
                            min(general_extent[0], ext[0]),
                            min(general_extent[1], ext[1]),
                            max(general_extent[2], ext[2]),
                            max(general_extent[3], ext[3])
                            )
                    
                    # Create bound polygon
                    poly_geom = Polygon([
                        (general_extent[0], general_extent[1]), 
                        (general_extent[2], general_extent[1]), 
                        (general_extent[2], general_extent[3]), 
                        (general_extent[0], general_extent[3])
                        ])
                    bound_poly = geopandas.GeoDataFrame(index=[0], geometry=[poly_geom])
                    bound_poly.crs = {'init': crs} 

                    return general_extent, bound_poly

            # Other data types
            else:
                raise ValueError('Input only supports geotif (tif) and shapefile (shp)')
        
        # Other cases
        else:
            raise ValueError('Checking file extension consistency\nInput have different data extensions')
        

# =========================================================================================== #
#              Convert meter to acr-degree based on latitude
# =========================================================================================== #
def meter2degree(input, latitude=None):
    """Convert image resolution from meter to acr-degree depending on location of latitude

    Args:
        input (numeric): Input resolution of distance
        latitude (numeric, optional): Latitude presents location. If latitude is None, the location is assumed near Equator. Defaults to None.

    Returns:
        Degree (float): Degree corresponding to the distance length

    """
    import numpy as np

    if latitude is None:
        # Equator location
        degree = input / (111320 * np.cos(np.radians(0.0)))
    else:
        degree = input / (111320 * np.cos(np.radians(latitude)))
    
    return degree


# =========================================================================================== #
#              Convert distance from degrees to meters depending on latitude
# =========================================================================================== #
def degree2meter(input, latitude=None):
    """Convert distance from degrees to meters depending on latitude

    Args:
        input (numeric): Input resolution in degrees
        latitude (numeric, optional): Latitude of the location. If latitude is None, the location is assumed near the Equator. Defaults to None.

    Returns:
        Distance length (numeric | float): Distance in meters corresponding to the input degree

    """
    import numpy as np

    if latitude is None:
        # Equator location
        meters = input * (111320 * np.cos(np.radians(0.0)))
    else:
        meters = input * (111320 * np.cos(np.radians(latitude)))
    
    return meters

# =========================================================================================== #
#              Computes the center latitude and longitude of a given geospatial input.
# =========================================================================================== #
def center_scene(input):
    """
    Computes the center latitude and longitude of a given geospatial input.

    Args:
        input (raster | shapefile): A geospatial object with a 'bounds' attribute that defines the spatial extent.

    Returns:
        tuple: A tuple containing:
            - center_lat (float): The center latitude of the input.
            - center_lon (float): The center longitude of the input.

    """
    import rasterio
    import geopandas

    # Define boundary 
    bounds, _ = get_extent_local(input)
    min_lon, min_lat, max_lon, max_lat = bounds[0], bounds[1], bounds[2], bounds[3]
    
    # Compute center latitude and longitude
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    return center_lat, center_lon


# =========================================================================================== #
#              Return min and max values of array or raster
# =========================================================================================== #
def mimax(input, digit=3):
    """Calculate maximum and minimum values of raster or array

    Args:
        input (raster | array): Raster image or data array
        digit (int, optional): Precise digit number. Defaults to 3.

    Returns:
        Min and Max values (numeric): Return 2 numbers of minvalue and maxvalue

    """
    import rasterio
    import numpy as np

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

    print(f"[Min: {min_round}  | Max: {max_round}]")

    return minValue, maxValue

# =========================================================================================== #
#              Calculate unique pixel values from a raster image or numpy array
# =========================================================================================== #
def unique_value(input, frequency=True, sort: Optional[AnyStr]='frequency'):
    """
    Calculate unique pixel values from a raster image or numpy array, optionally with their frequencies, and sort them.

    Args:
        input (raster| array): Input raster image or numpy array.
        frequency (bool, optional): If True, return the frequency of each unique value. Defaults to True.
        sort (str, optional): Sorting method for the unique values. Options are 'frequency' or 'value'. Defaults to 'frequency'.

    Returns:
        array or dataframe: Array with unique value if frequency is False, otherwise it returns DataFrame with unique values and frequencies.
    
    """
    import numpy as np
    import pandas as pd
    import rasterio
    from .processor import values

    # Check input data
    if isinstance(input, rasterio.DatasetReader):
        dataset = input.read()
    elif isinstance(input, np.ndarray):
        dataset = input
    else:
        raise ValueError('Input data is not supported')
    
    # Extract all values from raster or array
    pixel_values = values(dataset, na_rm=True)

    # Generate frequency and return
    if frequency is False:
        unique_values = np.sort(np.unique(pixel_values.values.flatten()))
    else:
        unique_values = pd.Series(pixel_values.values.ravel()).value_counts().reset_index()
        unique_values.columns = ['Value', 'Frequency']
        if sort.lower() == "frequency" or sort.lower() == "f":
            unique_values = unique_values.sort_values(by='Frequency')
        elif sort.lower() == "values" or sort.lower() == "value" or sort.lower() == "v":
            unique_values = unique_values.sort_values(by='Value')
        else:
            raise ValueError('Sort method is not supported ["frequency", "value]')
        
    return unique_values


# =========================================================================================== #
#              Convert a numpy array and metadata to a rasterio object stored in local variable
# =========================================================================================== #
def array2raster(array, metadata: Dict):
    """
    Convert a numpy array and metadata to a rasterio object stored in memory.

    Args:
        array (array): The input data array.
        metadata (Dict): The metadata dictionary.

    Returns:
        Local raster file (raster): The rasterio object stored in memory.

    """
    import rasterio

     # Determine number of bands
    if len(array.shape) == 3:
        nbands = array.shape[0]
    else:
        nbands = 1

    # Update metadata with the correct dtype and count
    metadata.update({
        'dtype': array.dtype,
        'count': nbands
    })

    # Write image in memory file and read it back
    memory_file = rasterio.MemoryFile()
    dst = memory_file.open(**metadata)

    if array.ndim == 2:
        dst.write(array, 1)
    elif array.ndim == 3:
        for i in range(array.shape[0]):
            dst.write(array[i, :, : ], i + 1)
    dst.close()

    # Read the dataset from memory
    dataset_reader = rasterio.open(dst.name, mode="r")

    return dataset_reader
    

# =========================================================================================== #
#              Reshapes a 3-dimensional numpy array between 'image' and 'raster' formats.
# =========================================================================================== #
def  reshape_raster(inputArray, mode:str="image"):
    """
    Reshapes a 3-dimensional numpy array between 'image' and 'raster' formats.

    Parameters:
        inputArray (array): The input 3-dimensional array to be reshaped.
        mode (str): The mode to reshape the array to. 'image' or 'img' reshapes to (height, width, bands), 'raster' or 'r' reshapes to (bands, height, width). Default is 'image'.

    Returns:
        Reshape array (array): The reshaped array.

    """
    import numpy as np

    # Check whether input are 3-dim data array
    if len(inputArray) == 3:
        # Convert to image
        if mode.lower() == 'image' or mode.lower() == 'img':
            output = np.transpose(inputArray, (1,2,0))
        # Convert to raster
        elif mode.lower() == 'raster' or mode.lower() == 'r':
            output = np.transpose(inputArray, (2,0,1))

        return output 
    
    #Other cases
    else:
        raise ValueError('Input data array must have 3 dimensions')
    
    