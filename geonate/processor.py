# Python 3.11.6
"""
The processor module

"""
# import common packages 
from typing import AnyStr, Dict, Optional

##############################################################################################

# =========================================================================================== #
#               Stack layer of geotif images
# =========================================================================================== #
def layestack(input):
    """
    Stacks multiple raster files or rasterio DatasetReader objects into a single multi-band raster.

    Parameters:
        input (list): List of file paths to the input raster files or rasterio DatasetReader objects.

    Returns:
        Stacked raster (raster): Stacked raster image.

    """
    import numpy as np
    from .geonate import rast
    from .common import array2raster, check_datatype_consistency, check_extension_consistency

    # Initialize some parameters and variables
    file2stack = []
    stacked_array = []
    nbands = len(input)

    consistency, datatype = check_datatype_consistency(input)

    # If input is list of file paths
    if (consistency is True) and datatype == "<class 'str'>":
        consistency_ext, extension = check_extension_consistency(input)
        
        # If input is a list of tif files
        if (consistency_ext is True) and (extension == 'tif'):
            # Stack each band 
            for i, bandi in enumerate(input):
                tmp = rast(input[i])
                ds = tmp.read(1) # Read each raster and read data array
                meta = tmp.meta 
                file2stack.append(ds) # stack each band in a list of data
        # Other data extension
        else:
            raise ValueError('Data type is not supported')
    
    # If input is local raster files 
    elif (consistency is True) and datatype == "<class 'rasterio.io.DatasetReader'>":
        # Stack each band 
        for i, bandi in enumerate(input):
            ds = input[i].read(1) # Read each band
            meta = bandi.meta 
            file2stack.append(ds) # stack each band in a list of data
    else:
        raise ValueError('Data type is not supported')    

    # convert list to array and update nbands
    stacked_array = np.stack(file2stack, axis=0) 
    meta.update({'count': nbands})

    # Convert array to raster
    stacked_image = array2raster(stacked_array, meta)

    return stacked_image


# =========================================================================================== #
#               Merge  geotif files in a list using GDAL and VRT
# =========================================================================================== #
def mergeVRT(input: AnyStr, output: AnyStr, compress: bool=True, silent=True):
    """Merge multiple geotif files using gdal VRT for better performance speed

    Args:
        input (list): List of input geotif files
        output (AnyStr): Path of output tif file
        compress (bool, optional): Whether compress the output data or not. Defaults to True.
        silent (bool, optional): Show or do not show file processing log. Defaults to True.
    
    Return:
        None: The function does not return any local variable. It writes raster file to local drive.

    """
    import os
    from osgeo import gdal
    #  Create a temp vrt file
    vrt_file = 'merged.vrt'

    if compress is True:
        vrt_options = gdal.BuildVRTOptions()
        gdal.BuildVRT(vrt_file, input, options=vrt_options)
        gdal.Translate(output, vrt_file, format='GTiff', creationOptions=['COMPRESS=LZW'])
        
    else:
        gdal.BuildVRT(vrt_file, input)
        gdal.Translate(output, vrt_file)
    
    os.remove(vrt_file)
    if silent is True:
        pass
    else:
        print(f"Finished merge raster files, the output is at {output}")
    

# =========================================================================================== #
#               Merge  geotif files in a list using Rasterio
# =========================================================================================== #
def merge(input: list):
    """
    Merges multiple raster files into a single raster file by computing the average values at overlapped areas.

    Args:
        input (list): List of input raster files.

    Returns:
        A merged raster (raster): The merged raster file.

    """
    from rasterio import merge 
    from .common import array2raster

    # Initialize empty list to store all input files and stack input into it 
    merged_files = []
    for tmp in input:
        merged_files.append(tmp)

    # Compute sum and count numbers of images, and average values at overlapped areas
    mosaic_sum, out_trans = merge.merge(merged_files, method= merge.copy_sum)
    mosaic_count, out_trans = merge.merge(merged_files, method= merge.copy_count)
    mosaic_average = mosaic_sum / mosaic_count

    # Update metadata with new transform and image dimensions
    meta = merged_files[0].meta
    meta.update({"driver": "GTiff",
                            "height": mosaic_average.shape[1],
                            "width": mosaic_average.shape[2],
                            "transform": out_trans})

    # Convert array to raster file
    merged_raster = array2raster(mosaic_average, meta)

    return merged_raster    


# =========================================================================================== #
#              Crop raster using shapefile or another image
# =========================================================================================== #
def crop(input, reference, invert=False, nodata=True):
    """
    Crops a raster file based on a reference shapefile or raster file. Optionally inverts the crop.

    Args:
        input (raster): The input raster file.
        reference (shapefile | raster): The reference shapefile (GeoDataFrame) or raster file (DatasetReader) to define the crop boundary.
        invert (bool, optional): If True, inverts the crop to mask out the area within the boundary. Defaults to False.
        nodata (bool, optional): If True, handles nodata values by converting the input to float32 and setting nodata to NaN. Defaults to True.

    Returns:
        A clipped raster (raster): The cropped raster file.
        
    """
    import rasterio
    import geopandas as gpd
    import numpy as np
    from rasterio.transform import Affine
    from shapely.geometry import mapping
    from shapely.geometry import box
    from .common import array2raster
    
    # Condition to process nodata
    if nodata is True:
        # Convert datatype of input to float32 to store NA value
        arr = input.read().astype(np.float32)
        meta = input.meta
        meta.update({'dtype': np.float32})
        input_image = array2raster(arr, meta)
    else: 
        input_image = input

    ### Define boundary
    # Reference is shapefile
    if isinstance(reference, gpd.GeoDataFrame):
        minx, miny, maxx, maxy = reference.total_bounds
        # define box
        bbox = box(minx, miny, maxx, maxy)
        poly_bound = gpd.GeoDataFrame({'geometry': [bbox]}, crs=reference.crs)

    # Reference is raster
    elif isinstance(reference, rasterio.DatasetReader):
        minx, miny, maxx, maxy = reference.bounds
        # define box
        bbox = box(minx, miny, maxx, maxy)
        poly_bound = gpd.GeoDataFrame({'geometry': [bbox]}, crs=reference.crs)

    # Others
    else:
        raise ValueError('Reference data is not supported')   

    ### Invert crop
    #### Condition for nodata
    if nodata is True:
        if invert is True:
            clipped, geotranform = rasterio.mask.mask(dataset=input_image, shapes= poly_bound.geometry.apply(mapping), invert=True, nodata= np.nan)
        else:
            clipped, geotranform = rasterio.mask.mask(dataset=input_image, shapes= poly_bound.geometry.apply(mapping), crop=True, invert=False, nodata= np.nan)
        
        # Update metadata
        meta  = input.meta
        meta.update({
            'height': clipped.shape[1],
            'width': clipped.shape[2],
            'transform': geotranform,
            'dtype': np.float32,
            'nodata': np.nan
            })
    #
    else:
        if invert is True:
            clipped, geotranform = rasterio.mask.mask(dataset=input_image, shapes= poly_bound.geometry.apply(mapping), invert=True, nodata= 0)
        else:
            clipped, geotranform = rasterio.mask.mask(dataset=input_image, shapes= poly_bound.geometry.apply(mapping), crop=True, invert=False, nodata= 0)
        
        # Update metadata
        meta  = input.meta
        meta.update({
            'height': clipped.shape[1],
            'width': clipped.shape[2],
            'transform': geotranform,
            'nodata': 0
            })
   
    # Convert array to raster
    clipped_raster = array2raster(clipped, meta)

    return clipped_raster

# =========================================================================================== #
#              Mask raster using shapefile or another image
# =========================================================================================== #
def mask(input, reference, invert=False, nodata=True):
    """ 
    Masks a raster file based on a reference shapefile or raster file. Optionally inverts the mask.

    Args:
        input (raster): The input raster file.
        reference (shapefile | raster): The reference shapefile (GeoDataFrame) or raster file (DatasetReader) to define the mask boundary.
        invert (bool, optional): If True, inverts the mask to mask out the area within the boundary. Defaults to False.
        nodata (bool, optional): If True, handles nodata values by converting the input to float32 and setting nodata to NaN. Defaults to True.

    Returns:
        A clipped and masked raster (raster): The cropped and masked raster file.
        
    """
    import numpy as np
    import rasterio
    import shapely
    from shapely.geometry import mapping
    import geopandas as gpd
    from .common import array2raster

    ##########################################
    #### Define boundary 
    if (isinstance(reference, gpd.GeoDataFrame)):
        poly = reference
        transform_poly = reference.transform
        crs_poly = reference.crs

    # Raster format
    elif isinstance(reference, rasterio.DatasetReader):
        ds_reference = reference.read(1)                                        # Extract only first band, transform, and crs
        transform_poly = reference.meta['transform']
        crs_poly = reference.meta['crs']

        # Create mask from all value different from Nodata
        masked = np.where(np.isnan(ds_reference), np.nan, 1)            # Replace values different than NA by 1
        masked_convert = masked.astype(np.float32)                             # Redefine data type of float32 to store n.nan
        
        # Create generator that yields (geometry, value) pairs for each shape found in the masked_convert
        shp = rasterio.features.shapes(masked_convert, mask= ~np.isnan(masked_convert), transform= transform_poly)
        poly = []
        values = []

        # Iterate over mask = 1, convert to shapely geometry and append 
        for shape, value in shp:
            if value == 1:
                poly.append(shapely.geometry.shape(shape))                         # convert a GeoJSON-like dictionary object into a Shapely geometry object
                values.append(value)
        
        # Create poly from mask
        poly = gpd.GeoDataFrame({'geometry': poly, 'value': values})
        poly.set_crs(crs_poly.to_string(), inplace=True)

    else:
        raise ValueError('Reference data is not supported')
    
    ##########################################
    ### Define nodata
    if nodata is True:
        # Convert datatype of input to float32 to store NA value
        arr = input.read().astype(np.float32)
        meta = input.meta
        meta.update({'dtype': np.float32})
        input_image = array2raster(arr, meta)

        ### Invert mask
        if invert is True:
            masked_img, geotranform = rasterio.mask.mask(dataset=input_image, shapes= poly.geometry.apply(mapping), crop=True, invert=True, nodata=np.nan)
        else:
            masked_img, geotranform = rasterio.mask.mask(dataset=input_image, shapes= poly.geometry.apply(mapping), crop=True, nodata= np.nan)

        meta  = input_image.meta
        meta.update({
            'height': masked_img.shape[1],
            'width': masked_img.shape[2],
            'transform': geotranform,
            'dtype': np.float32,
            'nodata': np.nan})

    else: 
        input_image = input

        ### Invert mask
        if invert is True:
            masked_img, geotranform = rasterio.mask.mask(dataset=input_image, shapes= poly.geometry.apply(mapping), crop=True, invert=True, nodata= 0)
        else:
            masked_img, geotranform = rasterio.mask.mask(dataset=input_image, shapes= poly.geometry.apply(mapping), crop=True, nodata= 0)

        meta  = input_image.meta
        meta.update({
            'height': masked_img.shape[1],
            'width': masked_img.shape[2],
            'transform': geotranform,
            'nodata': 0})

    ### Convert array back to raster         
    masked_raster = array2raster(masked_img, meta)
    
    return masked_raster


# =========================================================================================== #
#              Preprojection raster image 
# =========================================================================================== #
def reproject(input, reference, method: Optional[AnyStr]='near', res: Optional[float]=None):
    """
    Reprojects and resamples a given raster image to a specified coordinate reference system (CRS) and resolution.

    Args:
        input (raster): The input raster image to be reprojected.
        reference (raster | shapefile): The reference CRS for reprojection. It can be a string (e.g., 'EPSG:4326') or a rasterio DatasetReader object. If None, the input CRS is used.
        method (AnyStr, optional): The resampling method to use. Default is 'near'. Supported methods include 'nearest', 'average', 'max', 'min', 'median', 'mode', 'q1', 'q3', 'rms', 'sum', 'cubic', 'cubic_spline', 'bilinear', 'gauss', 'lanczos'.
        res (numeric, optional): The output resolution. If None, the input resolution is used.

    Returns:
        raster: The reprojected raster image
        
    """
    import numpy as np
    import rasterio
    from rasterio import warp
    from .common import array2raster
    
    # *********************************************
    # Define input image
    input_image = input.read()
    meta = input.meta
    left, bottom, right, top = input.bounds
    
    # *********************************************
    # Determine parameters and new transform
    # Reference string of EPSG
    if isinstance(reference, str):
        dst_crs = reference
        if res is None:
            raise ValueError('Please provide output resolution')
        else:
            xsize, ysize = res, res
        # Transform to new transform
        transform_new, width_new, height_new = warp.calculate_default_transform(src_crs=meta['crs'], dst_crs=dst_crs, \
                                                                                                                                                height=meta['height'], width=meta['width'], \
                                                                                                                                                resolution=(xsize, ysize), \
                                                                                                                                                left=left, bottom=bottom, right=right, top=top)
    # Take all paras from reference image
    elif isinstance(reference, rasterio.DatasetReader):
        dst_crs = reference.crs
        if res is None:
            xsize, ysize = reference.res
        else:
            xsize, ysize = res, res
        # Transform to new transform
        transform_new, width_new, height_new = warp.calculate_default_transform(src_crs=meta['crs'], dst_crs=dst_crs, \
                                                                                                                                                height=meta['height'], width=meta['width'], \
                                                                                                                                                resolution=(xsize, ysize), \
                                                                                                                                                left=left, bottom=bottom, right=right, top=top)
    # Other cases
    else:
        raise ValueError('Please define correct reference, it is CRS string or an image reference')

    # *******************************************
    # Update metadata
    meta_update = meta.copy()
    meta_update.update({
        'crs': dst_crs,
        'transform': transform_new,
        'width': width_new,
        'height': height_new,
    })
    # *******************************************
    # Resampling method
    if method.lower() == 'near' or method.lower() == 'nearest':
        resampleAlg = warp.Resampling.nearest
    elif method.lower() == 'mean' or method.lower() == 'average':
        resampleAlg = warp.Resampling.average
    elif method.lower() == 'max':
        resampleAlg = warp.Resampling.max
    elif method.lower() == 'min':
        resampleAlg = warp.Resampling.min
    elif (method.lower() == 'median') or (method.lower() == 'med'):
        resampleAlg = warp.Resampling.med
    elif method.lower() == 'mode':
        resampleAlg = warp.Resampling.mode
    elif method.lower() == 'q1':
        resampleAlg = warp.Resampling.q1
    elif method.lower() == 'q3':
        resampleAlg = warp.Resampling.q3
    elif method.lower() == 'rsm':
        resampleAlg = warp.Resampling.rms
    elif method.lower() == 'sum':
        resampleAlg = warp.Resampling.sum
    elif method.lower() == 'cubic':
        resampleAlg = warp.Resampling.cubic
    elif method.lower() == 'spline':
        resampleAlg = warp.Resampling.cubic_spline
    elif method.lower() == 'bilinear':
        resampleAlg = warp.Resampling.bilinear
    elif method.lower() == 'gauss':
        resampleAlg = warp.Resampling.gauss
    elif method.lower() == 'lanczos':
        resampleAlg = warp.Resampling.lanczos
    else:
        raise ValueError('The resampling method is not supported, available methods rasterio.warp.Resampling')

    # ***************************************
    # Running reproject 
    projected_array = np.empty((input_image.shape[0], height_new, width_new), dtype= meta['dtype'])
    for band in range(0, input_image.shape[0]):
        ds = input_image[band, : , : ]
        warp.reproject(source=ds, destination=projected_array[(band), :, :], \
                       src_transform= meta['transform'], dst_transform=transform_new, \
                        src_crs=meta['crs'], dst_crs=dst_crs, \
                        resampling= resampleAlg)
    
    # *****************************************
    # Convert array back to raster
    reprojected = array2raster(projected_array, meta_update)
    
    return reprojected


# =========================================================================================== #
#              Resample raster image based on factor
# =========================================================================================== #
def resample(input, factor, mode='aggregate', method='near'):
    """
    Resample raster image based on factor

    Args:
        input (DatasetReader): Input rasterio image.
        factor (numeric): Resampling factor compared to original image (e.g., 2, 4, 6).
        mode (str, optional): Resample mode ["aggregate", "disaggregate"]. Defaults to 'aggregate'.
        method (str, optional): Resampling method (e.g., 'nearest', 'cubic', 'bilinear', 'average'). Defaults to 'near'.

    Returns:
        raster: Resampled raster image.        

    """
    import rasterio
    from rasterio import warp
    import numpy as np
    from .common import array2raster

    # *****************************************
    ### Define input image
    # input is raster
    if isinstance(input, rasterio.DatasetReader):
        dataset = input.read()
        meta = input.meta
        left, bottom, right, top = input.bounds
        nbands = input.count
    # Other input
    else:
        raise ValueError('Input data is not supported')
    
    # *****************************************
    #### Calculate new rows and columns
    if (mode.lower() == 'aggregate') or (mode.lower() == 'agg') or (mode.lower() == 'a'):
        new_height = meta['height'] // factor
        new_width = meta['width'] // factor

    elif (mode.lower() == 'disaggregate') or (mode.lower() == 'disagg') or (mode.lower() == 'd'):
        new_height = meta['height'] * factor
        new_width = meta['width'] * factor

    else:
        raise ValueError('Resample method is not supported ["aggregate", "disaggregate"]')

    # *****************************************
    # Calculate new transform
    transform_new, width, height = warp.calculate_default_transform(src_crs=meta['crs'], dst_crs=meta['crs'], width=new_width, height=new_height, left=left, bottom=bottom, right=right, top=top)

    # *****************************************
    # Resampling method
    if method.lower() == 'near' or method.lower() == 'nearest':
        resampleAlg = warp.Resampling.nearest
    elif method.lower() == 'mean' or method.lower() == 'average':
        resampleAlg = warp.Resampling.average
    elif method.lower() == 'max':
        resampleAlg = warp.Resampling.max
    elif method.lower() == 'min':
        resampleAlg = warp.Resampling.min
    elif (method.lower() == 'median') or (method.lower() == 'med'):
        resampleAlg = warp.Resampling.med
    elif method.lower() == 'mode':
        resampleAlg = warp.Resampling.mode
    elif method.lower() == 'q1':
        resampleAlg = warp.Resampling.q1
    elif method.lower() == 'q3':
        resampleAlg = warp.Resampling.q3
    elif method.lower() == 'rsm':
        resampleAlg = warp.Resampling.rms
    elif method.lower() == 'sum':
        resampleAlg = warp.Resampling.sum
    elif method.lower() == 'cubic':
        resampleAlg = warp.Resampling.cubic
    elif method.lower() == 'spline':
        resampleAlg = warp.Resampling.cubic_spline
    elif method.lower() == 'bilinear':
        resampleAlg = warp.Resampling.bilinear
    elif method.lower() == 'gauss':
        resampleAlg = warp.Resampling.gauss
    elif method.lower() == 'lanczos':
        resampleAlg = warp.Resampling.lanczos
    else:
        raise ValueError('The resampling method is not supported, available methods raster.Resampling.')

    # *****************************************
    # Define and Update the metadata for the destination raster
    metadata = meta.copy()
    metadata.update({
        'transform': transform_new,
        'width': new_width,
        'height': new_height, 
        'dtype': np.float32
    })

    # *****************************************
    # Run Resampling for each band
    resampled = np.empty((nbands, new_height, new_width), dtype=np.float32)

    for band in range(0, nbands):
        if nbands <= 1:
            ds = dataset
        else:
            ds = dataset[band, : , : ]

        warp.reproject(source=ds, destination=resampled[band, :, :], \
                       src_transform= meta['transform'], dst_transform= transform_new, \
                        src_crs=meta['crs'], dst_crs=input.crs, resampling= resampleAlg)

    # *****************************************
    # Convert array to raster 
    resampled_raster = array2raster(resampled, metadata)

    return resampled_raster

# =========================================================================================== #
#              Matching two images to have the same boundary
# =========================================================================================== #
def match(input, reference, method='near'):
    """
    Match input image to the reference image in terms of projection, resolution, and bound extent. It returns image within the bigger boundary.

    Args:
        input (raster): Rasterio objective needs to match the reference.
        reference (raster): Rasterio object taken as reference to match the input image.
        method (AnyStr, optional): String defines resampling method (if applicable) to resample if having different resolution (Method similar to resample). Defaults to 'near'.

    Returns:
        raster: Matched raster image with the same projection, resolution, and extent as the reference image.

    """
    import rasterio
    from rasterio import warp
    from rasterio.transform import from_bounds
    import numpy as np
    from .common import get_extent_local, array2raster
    
    # *****************************************
    ### Define input image
    # input is raster
    if isinstance(input, rasterio.DatasetReader):
        input_image = input.read()
        meta = input.meta
    # Other input
    else:
        raise ValueError('Input data is not supported')
    
    # *****************************************
    ### Define reference image
    if isinstance(reference, rasterio.DatasetReader):
        reference_image = reference.read()
        meta_reference = reference.meta
    # Other input
    else:
        raise ValueError('Input data is not supported')
    
    # *****************************************
    ### Check CRS and Resolution
    if (meta["crs"] != meta_reference['crs']) or (meta['transform'][0] != meta_reference['transform'][0]):
        raise ValueError('Input and reference images have different Projection and Resolution')
    # If having the same CRS and Resolution
    else:
        
        # *****************************************
        # Get general extent from two images
        ext_input = get_extent_local(input)[0]
        ext_reference = get_extent_local(reference)[0]
        
        ext = ext_input
        ext = (
            min(ext[0], ext_reference[0]),
            min(ext[1], ext_reference[1]),
            max(ext[2], ext_reference[2]),
            max(ext[3], ext_reference[3])
            )
        
        # *****************************************
        # Calculate new height & width and new transform
        resolution = meta_reference['transform'][0]    
        width_new = int((ext[2]  - ext[0]) / resolution)
        height_new = int((ext[3] - ext[1]) / resolution)
    
        transform_new = from_bounds(ext[0], ext[1], ext[2], ext[3], width_new, height_new)
        
        # *****************************************
        # Resampling method
        if method.lower() == 'near' or method.lower() == 'nearest':
            resampleAlg = warp.Resampling.nearest
        elif method.lower() == 'mean' or method.lower() == 'average':
            resampleAlg = warp.Resampling.average
        elif method.lower() == 'max':
            resampleAlg = warp.Resampling.max
        elif method.lower() == 'min':
            resampleAlg = warp.Resampling.min
        elif (method.lower() == 'median') or (method.lower() == 'med'):
            resampleAlg = warp.Resampling.med
        elif method.lower() == 'mode':
            resampleAlg = warp.Resampling.mode
        elif method.lower() == 'q1':
            resampleAlg = warp.Resampling.q1
        elif method.lower() == 'q3':
            resampleAlg = warp.Resampling.q3
        elif method.lower() == 'rsm':
            resampleAlg = warp.Resampling.rms
        elif method.lower() == 'sum':
            resampleAlg = warp.Resampling.sum
        elif method.lower() == 'cubic':
            resampleAlg = warp.Resampling.cubic
        elif method.lower() == 'spline':
            resampleAlg = warp.Resampling.cubic_spline
        elif method.lower() == 'bilinear':
            resampleAlg = warp.Resampling.bilinear
        elif method.lower() == 'gauss':
            resampleAlg = warp.Resampling.gauss
        elif method.lower() == 'lanczos':
            resampleAlg = warp.Resampling.lanczos
        else:
            raise ValueError('The resampling method is not supported, available methods raster.Resampling.')

        # *****************************************
        # Reproject to match
        if len(input_image.shape) > 2:
            nbands = input_image.shape[0]
        else:
            nbands = 1
        # Run over each band
        matched = np.empty((nbands, height_new, width_new), dtype=np.float32)
        for band in range(0, nbands):
            if nbands <= 1:
                ds = input_image
            else:
                ds = input_image[band, : , : ]
            warp.reproject(source=ds, destination=matched[band, :, :], src_transform= meta['transform'], dst_transform= transform_new, src_crs=meta['crs'], dst_crs=meta_reference['crs'], resampling= resampleAlg)
        
        # *****************************************
        # Mask out other values
        match_masked = np.where(matched == 0, np.nan, matched)
        match_masked = match_masked.astype(np.float32)

        # *****************************************
        # Update metadata and Convert to raster
        meta_update = meta.copy()
        meta_update.update({
            'crs': meta_reference['crs'],
            'transform': transform_new,
            'width': width_new,
            'height': height_new,
            'dtype': np.float32
        })
        match_raster = array2raster(match_masked, meta_update)
        
        return match_raster
   

# =========================================================================================== #
#              Calculate normalized difference index 
# =========================================================================================== #
def normalizedDifference(input, band1, band2):
    """
    Calculate normalized difference index

    Args:
        input (Raster | Array): Rasterio object or data array, input with multiple bands.
        band1 (numeric): Order of the first band in the input.
        band2 (numeric): Order of the second band in the input.

    Returns:
        raster | dataArray: Normalized difference result in raster or data array depending on input, containing all image pixel values

    """
    import numpy as np
    import rasterio
    from .common import array2raster

    # *****************************************
    # Define data input
    # input is raster
    if isinstance(input, rasterio.DatasetReader):
        dataset = input.read()
        meta = input.meta
    # input is array
    elif isinstance(input, np.ndarray):
        dataset = input
        meta = None
    # Other input
    else:
        raise ValueError('Input data is not supported')

    # *****************************************
    # Extract band values
    ds_band1 = dataset[band1+1, : , : ]
    ds_band2 = dataset[band2+1, : , : ]

    # *****************************************
    # Calculate index and Remove outliers, also
    normalized_index  = (ds_band1.astype(float) - ds_band2.astype(float)) / (ds_band1 + ds_band2)
    normalized_index = normalized_index.astype(np.float32)

    normalized_index[(normalized_index < -1) | (normalized_index > 1)] = np.nan

    # *****************************************
    # Define output 
    if isinstance(input, rasterio.DatasetReader):
        meta.update({'dtype': np.float32, 'count': 1})                                                  #  update datatype in metadata 
        normalized_output = array2raster(normalized_index, meta)
    elif isinstance(input, np.ndarray):
        normalized_output = normalized_index
    else:
        normalized_output = []

    return normalized_output


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
#              Extract raster values of all bands and create dataframe
# =========================================================================================== #
def values(input, na_rm: Optional[bool]=True, names: Optional[list]=None, prefix: Optional[AnyStr]=None):
    """
    Extract all pixel values of image and create dataframe from them, each band is a column

    Args:
        input (raster | array): Rasterio raster image or data array.
        na_rm (bool, optional): Remove or do not remove NA value from output dataframe. Defaults to True.
        names (list, optional): Given expected names for each column in the dataframe, if not, default name will be assigned. Defaults to None.
        prefix (AnyStr, optional): Given character before each band name. Defaults to None.

    Returns:
        DataFrame: Dataframe stores all pixel values across all image bands.

    """
    import rasterio
    import numpy as np
    import pandas as pd

    # *****************************************
    # Check input data
    # input is raster
    if isinstance(input, rasterio.DatasetReader):
        dataset = input.read()
    # input is array
    elif isinstance(input, np.ndarray):
        dataset = input
    # Other input
    else:
        raise ValueError('Input data is not supported')
    
    # *****************************************
    # Define parameters
    nbands = dataset.shape[0]
    bands_array = [dataset[band, : , : ].flatten() for band in range(0, nbands)]
    
    # *****************************************
    # Assign column names in case name is given 
    if names is not None:
        if len(names) != nbands:
            raise ValueError('Length of name should be equal to number of bands')
        else:
            if prefix is None:
                data = pd.DataFrame(np.array(bands_array).T, columns=names)
            else:
                names_new = [f'{prefix}{name}' for name in names]
                data = pd.DataFrame(np.array(bands_array).T, columns=names_new)
    # If name is not given
    else:
        if prefix is None:
            data = pd.DataFrame(np.array(bands_array).T, columns=[f'B{i}' for i in range(1,nbands +1)])
        else:
            data = pd.DataFrame(np.array(bands_array).T, columns=[f'{prefix}{i}' for i in range(1, nbands +1)])
    
    # *****************************************
    # Remove NA values or not
    if na_rm is True: 
        data_out = data.dropna().reset_index(drop=True)
    else:
        data_out = data

    return data_out


 # =========================================================================================== #
#              Extract pixel values at GCP (points or polygon)
# =========================================================================================== #       
def extractValues(input, roi, field, dataframe: Optional[bool]=True, names: Optional[list]=None, prefix: Optional[AnyStr]=None, tail=True):
    """
    Extract pixel values from a raster image based on regions of interest (ROI) defined in a shapefile.

    Args:
        input (raster): Rasterio image as input
        roi (shapefile): Shapefile where GCP points are located, read by geopandas.
        field (AnyStr): Field name in shapefile GCP to extract label values, e.g., 'class'. **But this field must store number instead of string**.
        dataframe (bool, optional): Whether to return a dataframe or separate X, y arrays. Defaults to True.
        names (list, optional): Expected names for each column in the dataframe. Defaults to None.
        prefix (AnyStr, optional): Prefix for each band name. Defaults to None.
        tail (bool, optional): Whether to place the class value at the end or front of the dataframe. Defaults to True.

    Returns:
        DataFrame or Tuple: Dataframe if dataframe=True, otherwise X and y arrays for training a model.

    """
    import os
    import rasterio
    from rasterio.plot import reshape_as_image
    import numpy as np
    from shapely.geometry import mapping
    import pandas as pd
    from .common import array2raster

    # *****************************************
    # Define input image
    # Other data type
    if not isinstance(input, rasterio.DatasetReader):
        raise ValueError('Input data is not supported')
    # Input is raster
    else:
        # Convert datatype of input to float32 to store NA value
        arr = input.read().astype(np.float32)
        meta = input.meta
        meta.update({'dtype': np.float32})
        input_image = array2raster(arr, meta)
        
    # *****************************************
    # Convert shapefile to shapely geometry
    geoms = roi.geometry.values
    
    # Extract some metadata information
    nbands = input_image.count
    dtype_X = np.float32()
    dtype_y = np.float32()

    # Create empty array to contain X and y arrays
    X = np.array([], dtype= dtype_X).reshape(0, nbands)
    y = np.array([], dtype= dtype_y)

    # Run loop over each features in shapefile to extract pixel values
    for index, geom in enumerate(geoms):
        poly = [mapping(geom)]

        # Crop image based on feature
        cropped, transform = rasterio.mask.mask(input_image, poly, crop=True, nodata=np.nan)

        # Reshape dataset in form of (values, bands)
        cropped_reshape = reshape_as_image(cropped)
        reshapped = cropped_reshape.reshape(-1, nbands)

        # Append 1D array y
        y = np.append(y, [roi[field][index]] * reshapped.shape[0])
        
        # vertical stack 2D array X
        X = np.vstack((X, reshapped))
    
    # Remove NA value from data
    data = np.hstack((X, y.reshape(y.shape[0], 1))).astype(np.float32)
    data_na = data[~np.isnan(data).any(axis=1)]
    data_nodata = data_na[~(data_na == np.nan).any(axis=1)]

    X_na = data_nodata[ :, 0:nbands]
    y_na = data_nodata[ : , nbands]

    # return dataframe
    if dataframe is True:
        y_na_reshape = y_na.reshape(-1,1)

        # class tail
        if tail is True:
            arr = np.hstack([X_na, y_na_reshape])
        else:
            arr = np.hstack([y_na_reshape, X_na])
        
        # Name is not given
        if names is None:
            if prefix is None:
                names_band = [f'B{i}' for i in range(1, input_image.count +1)]
                name_class = [str(field)]
                if tail is True:
                    names_list = names_band + name_class
                else:
                    names_list = name_class + names_band
            else:
                names_band = [f'{prefix}{i}' for i in range(1, input_image.count +1)]
                name_class = [str(field)]
                if tail is True:
                    names_list = names_band + name_class
                else:
                    names_list = name_class + names_band
            data = pd.DataFrame(arr, columns=names_list)            
            return data
        
        # Name is given
        else:
            if len(names) != (nbands + 1):
                raise ValueError('Length of name should be equal to number of bands plus 1')
            else:
                if prefix is None:
                    names_list = names
                else:
                    names_list = [f'{prefix}{name_i}' for name_i in names]
            data = pd.DataFrame(arr, columns=names_list)            
            return data
    
    # Do not return dataframe
    else:
        return X_na, y_na
    

# =========================================================================================== #
#              Normalize raster data
# =========================================================================================== #      
def normalized(input):
    """
    Normalize raster data to rearrange raster values from 0 to 1

    Args:
        input (DatasetReader | np.ndarray): Rasterio image or data array.

    Returns:
        raster | data array: Data array or raster depends on the input files.

    """
    import rasterio
    import numpy as np
    import pandas as pd
    from .common import array2raster

    ### Check input data
    if isinstance(input, rasterio.DatasetReader):
        dataset = input.read()
        meta = input.meta
    elif isinstance(input, np.ndarray):
        dataset = input
        meta = meta
    else:
        raise ValueError('Input data is not supported')
    
    ### Find max min values
    maxValue = np.nanmax(dataset)
    minValue = np.nanmin(dataset)

    ### Create empty data array to store output
    normalized = np.zeros_like(dataset, dtype=np.float32)

    ### Run normalization for each 
    for i in range(0, dataset.shape[0]):
        band = dataset[i, : , : ]
        band_norm = (band.astype(float)  - minValue) / (maxValue  - minValue)
        normalized[i, : , : ] = band_norm
        band_norm = None        # set to None after the iteration

    ### Define output
    if isinstance(input, rasterio.DatasetReader):
        meta.update({'dtype': np.float32})
        normalized_raster = array2raster(normalized, meta)
    else:
        normalized_raster = normalized
    
    return normalized_raster
    

# =========================================================================================== #
#              Principal Component Analysis (PCA) on multispectral data for data Dimension Reduction.
# =========================================================================================== #      

def pca(input, n_component: int=3, **kwargs):
    """
    Perform Principal Component Analysis (PCA) on multispectral data for data Dimension Reduction.

    Args:
        input (rasterio.DatasetReader or np.ndarray): Multispectral input data. Can be a raster image or a numpy array.
        n_component (int): Number of principal components to compute. Default is 3.
        **kwargs (optional): All parameters in sklearn.decomposition.PCA()

    Returns:
        np.ndarray or rasterio.DatasetReader: PCA-transformed data in the same format as the input.

    """
    import numpy as np
    import rasterio
    from sklearn.decomposition import PCA
    from .common import array2raster, reshape_raster

    # Identify datatype and define input data
    # Raster image
    if isinstance(input, rasterio.DatasetReader):
        arr = input.read()
        height, width = input.shape
        nbands =  input.count
        meta = input.meta
    # Data Array
    elif isinstance(input, np.ndarray):
        if len(input.shape) < 3:
            raise ValueError('Input must be multispectral data (multi-band)')
        else:
            arr = input
            nbands, height, width = input.shape

    else: 
        raise ValueError('Input is not supported')
    
    # Reshape from raster to image format, and from 3D to 2D
    ds = reshape_raster(arr, mode='image')
    ds_reshaped = ds.reshape((-1, nbands)) # -1 means this dim will be determined by other dims

    # Define PCA model and fit the PCA model
    pca_model = PCA(n_components= n_component, **kwargs)
    pca_fit = pca_model.fit_transform(ds_reshaped, **kwargs)

    # Reshape the result to the new shape
    pca_img = pca_fit.reshape((height, width, n_component))
    
    # Reshape from image to raster format 
    pca_raster = reshape_raster(pca_img, mode='raster')

    # Return output based on input similar to input
    if isinstance(input, np.ndarray):
        return pca_raster
    
    # Convert to rasterio object
    elif isinstance(input, rasterio.DatasetReader):
        meta.update({'count': n_component})
        pca_rast = array2raster(pca_raster, meta)
        return pca_rast

