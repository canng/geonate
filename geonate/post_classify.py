"""
The Post-processing module

"""
# import common packages 
from typing import AnyStr, Dict, Optional

# =========================================================================================== #
#              Reclassify image
# =========================================================================================== #
def reclassify(input, breakpoints, classes):
    """
    Reclassify image with discrete or continuous values

    Args:
        input (raster | array): Raster or data array input
        breakpoints (list): Number list, defines a breakpoint value for reclassifcation, e.g., [ -1, 0, 1]
        classes (list): Number list, define classes, number of classes equal number of breakpoints minus 1

    Returns:
        raster | dataArray: Reclassified result in raster or data array depending on input, containing all image pixel values

    """
    import rasterio
    import numpy as np
    from .common import array2raster

    # *****************************************
    # Check input data
    # input is raster
    if isinstance(input, rasterio.DatasetReader):
        if len(input.shape) == 2:
            dataset = input.read()
            meta = input.meta
        elif len(input.shape) == 3:
            if  input.shape[0] > 1:
                raise ValueError('Input data has more than one band')
            else:
                dataset = input.read(1)
                meta = input.meta
    # input is array
    elif isinstance(input, np.ndarray):
        if (len(input.shape)) > 2 and (input.shape[0] > 1):
            raise ValueError('Input data has more than one band')
        else:
            dataset = input
    # Other input
    else:
        raise ValueError('Input data is not supported')

    # *****************************************
    # Create unique values and empty data array to store reclassified result 
    uniques = np.unique(dataset)
    reclassified = np.zeros_like(dataset)
        
    # *****************************************
    # If image has discrete values
    if len(uniques) == len(classes): 
        if len(breakpoints) == len(classes):
            for i in range(len(classes)):
                reclassified[dataset == breakpoints[i]] = classes[i]
        elif len(breakpoints) == (len(classes)-1):
            for i in range(len(classes)):
                reclassified[(dataset >= breakpoints[i]) & (dataset < breakpoints[i+1])] = classes[i]
        else:
            raise ValueError('Number of classes must be equal to number of breakpoints minus 1')
        
    # If image has continuous values
    else:
        if len(breakpoints) == (len(classes)+1):
            for i in range(len(classes)):
                reclassified[(dataset >= breakpoints[i]) & (dataset < breakpoints[i+1])] = classes[i]
        else:
            raise ValueError('Number of classes must be equal to number of breakpoints minus 1')
    
    # *****************************************
    # Define output
    if isinstance(input, rasterio.DatasetReader):
        reclassified_raster = array2raster(reclassified, meta)
    else:
        reclassified_raster = reclassified

    return reclassified_raster

# =========================================================================================== #
#              Converts a raster image to a shapefile
# =========================================================================================== #
def raster2shapefile(image, band=1):
    """
    Converts a raster image to a shapefile (GeoDataFrame).
    This function reads a raster image, masks out NoData values, extracts shapes (polygons) from the raster,
    and converts them into a GeoDataFrame.

    Args:
        image (rasterio.io.DatasetReader): The raster image to be converted.
        band (int, optional): The band number to read from the raster image. Defaults to 1.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the extracted polygons and their corresponding values.

    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape

    # Check input data
    # Other input
    if not isinstance(input, rasterio.DatasetReader):
        raise ValueError('Input data is not supported, it must be raster object')    
    
    # Other input
    else:
        ds =image.read(band)  
        transform = image.transform
        
        # Mask out NoData values
        mask = image != image.nodata
        
        # Extract shapes (polygons) from the raster
        shapes_gen = shapes(ds, mask=mask, transform=transform)
        
        # Convert to GeoDataFrame
        polygons = []
        values = []
        for geom, value in shapes_gen:
            polygons.append(shape(geom))
            values.append(value)
        
        gdf = gpd.GeoDataFrame({'geometry': polygons, 'class': values}, crs=image.crs)
        
        return gdf