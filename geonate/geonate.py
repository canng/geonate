"""
Main module.

"""
# import common packages 
from typing import AnyStr, Dict, Optional

##############################################################################################

# =========================================================================================== #
#               Open raster geotif file
# =========================================================================================== #
def rast(input: AnyStr, show_meta: Optional[bool]=False, **kwargs):
    """Open a single geotif raster file using Rasterio

    Args:
        input (AnyStr): The file path indicates location of geotif file
        show_meta (bool, optional): Whether to show the image metadata. Defaults to False.
        **kwargs (optional): All parameters in rasterio.open()

    Returns:
        Raster object (raster): Rasterio RasterReader object

    """    
    import rasterio
    import os

    img = rasterio.open(input, **kwargs)
    basename = os.path.basename(input)
    
    # show meta 
    if show_meta is True:
        meta = img.meta
        print(f"Opening: {basename}\n{meta}")
    
    return img    

 
# =========================================================================================== #
#               Open shapefile
# =========================================================================================== #
def vect(input: AnyStr, show_meta: Optional[bool]=False, **kwargs):
    """Read shapefile vector file using Geopandas 

    Args:
        input (AnyStr): The file path indicates location of shapefile 
        show_meta (bool, optional): Whether to show the image metadata. Defaults to False.
        **kwargs (optional): All parameters in gpd.read_file()

    Returns:
        Shapefile (geodataframe): Geodataframe of shapefile with attributes from geopandas object

    """
    import geopandas as gpd
    import os
    
    vect = gpd.read_file(input, **kwargs)

    # show meta 
    if show_meta is True:
        basename = os.path.basename(input)
        crs = vect.crs
        datashape = vect.shape
        print(f"Opening: {basename}\n Projection (crs): {crs}\n Data shape: {datashape}")

    return vect

# =========================================================================================== #
#               Compress file size and write geotif
# =========================================================================================== #
def writeRaster(input, output, meta: Optional[Dict]=None, compress: Optional[AnyStr] = 'lzw'):
    """Write raster Geotif from Raster or Data Array using Rasterio

    Args:
        input (raster | array): Raster or Data array in form of [band, height, width]
        output (AnyStr): Output file path
        meta (Dict, optional): Rasterio profile settings needed when input is dataArray. Defaults to None.
        compress (AnyStr, optional): Compression algorithm ['lzw', 'deflate']. Defaults to 'lzw'.

    Returns:
        None: The function does not return any local variable. It writes raster file to local drive (.tif).

    """   
    import rasterio
    import numpy as np
  
    # Input is rasterio image
    if isinstance(input, rasterio.DatasetReader):
        meta_out = input.meta
        data_array = input.read()

        # compress data or not
        if compress is None:
            meta_out = meta_out
        else:
            if compress.lower() == 'deflate':
                meta_out.update({'compress': 'deflate'})
            elif compress.lower() == 'lzw':
                meta_out.update({'compress': 'lzw'})
            else:
                raise ValueError('Compress method is not supported')

        # output has single band
        if len(data_array.shape) == 2:
            meta_out['count'] = int(1)
            with rasterio.open(output, 'w', **meta_out) as dst:
                for band in range(0, 1):
                    data = data_array
                    dst.write(data, band + 1)
        # output has multi bands
        else:
            meta_out['count'] = int(data_array.shape[0])
            with rasterio.open(output, 'w', **meta_out) as dst:
                for band in range(0, int(data_array.shape[0])):
                    data = data_array[band, : , : ]
                    dst.write(data, band + 1)

    # input is data array
    elif isinstance(input, np.ndarray):
        if meta is None:
            raise ValueError('Input is dataArray, please give metadata profile')
        else:        
        # compress data or not
            if compress is None:
                meta = meta
            else:
                if compress.lower() == 'deflate':
                    meta.update({'compress': 'deflate'})
                elif compress.lower() == 'lzw':
                    meta.update({'compress': 'lzw'})
                else:
                    raise ValueError('Compress method is not supported')

            # output has single band
            if len(input.shape) == 2:
                meta['count'] = int(1)
                with rasterio.open(output, 'w', **meta) as dst:
                    for band in range(0, 1):
                        data = input
                        dst.write(data, band + 1)
            # output has multi bands
            else:
                meta['count'] = int(input.shape[0])
                with rasterio.open(output, 'w', **meta) as dst:
                    for band in range(0, int(input.shape[0])):
                        data = input[band, : , : ]
                        dst.write(data, band + 1)
    else:
        raise ValueError('Input data is not supported')    


