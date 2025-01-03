# Python 3.11.6
from typing import AnyStr, Dict, Optional
import os
from rasterio.warp import Resampling

##############################################################################################
'''
RASTER FUNCTIONS

1. crop
2. extractValues
3. layerstack
4. mask
5. match
6. merge
7. mergeVRT
8. normalizedDifference
9. project
10. rast
11. reclassify
12. values
13. vect
14. writeRaster

'''
# =========================================================================================== #
#               Take plot function from earthpy
# =========================================================================================== #

import earthpy.plot as plot


# =========================================================================================== #
#               Open raster geotif file
# =========================================================================================== #
def rast(input: AnyStr, show_meta: Optional[bool]=False):
    '''
    Open a single geotif raster file using Rasterio

    Parameters:
        inputpath: the file path indicates location of geotif file

    Example:
       path = '../test/landsat_multi/landsat_img_test.tif'
       img = raster.rast(path)
    
    '''
    import rasterio
    import os

    img = rasterio.open(input)
    basename = os.path.basename(input)
    
    # show meta 
    if show_meta is True:
        meta = img.meta
        print(f"Opening: {basename} \n{meta}")
    
    return img    


# =========================================================================================== #
#               Get bounds of raster
# =========================================================================================== #
def getBounds(input: AnyStr, meta: Optional[Dict]=None):
    '''
    Return bounds of raster image

    Parameters:
        input: image or data array input that need to crop
        meta: Optional metadata when input is data array

    Example:
        img = raster.rast('./Sample_data/landsat_multi/landsat_stacked.tif')
        meta = img.meta
        ds= img.read()

        left, bottom, right, top = raster.getBounds(ds, meta)
        print(left, bottom, right, top)
    
    '''
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
#               Open shapefile
# =========================================================================================== #
def vect(input: AnyStr, show_meta: Optional[bool]=False):
    '''
    Read shapefile vector file using Geopandas 

    Parameters:
        input_path: the file path indicates location of shapefile 

    Example:
        path ='../test/roi/roi.shp
        poly = raster.vect(path)

    '''
    import geopandas as gpd
    import os
    
    vect = gpd.read_file(input)

    # show meta 
    if show_meta is True:
        basename = os.path.basename(input)
        crs = vect.crs
        datashape = vect.shape
        print(f"Opening: {basename} \n Projection (crs): {crs} \n Data shape: {datashape}")

    return vect

# =========================================================================================== #
#               Compress file size and write geotif
# =========================================================================================== #
def writeRaster(input: AnyStr, output: AnyStr, meta: Optional[Dict]=None, compress: Optional[AnyStr] = 'lzw'):
    '''
    Write raster Geotif from data Array using Rasterio.

    Parameters:
        input: Data array in form of [band, height, width]
        output: Output file path.
        meta: Rasterio profile settings.
        compress: Boolean indicating whether to compress the output.
        compress_opt: Compression algorithm (optional).

    Example:
        meta = img.meta
        meta.update({'count': 1})
        meta.update({'dtype': rasterio.float32})
        raster.writeRatser(arr, 'output.tif', meta=meta, compress='LZW')

    '''
    
    import rasterio
    import numpy as np

    class RasterWriteError(Exception):
        pass
    
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

# =========================================================================================== #
#               Stack layer of geotif images
# =========================================================================================== #
def layerstack(input, output: Optional[AnyStr]=None):
    '''
    Stack layers for different geotif images with the same extent and each image may have more than 1 band

    Parameters:
        input: String, List of input geotif files
        output: Optional string, whether to write the stacked image or not

    Example:
       input = tools.list_files('./Sample_data/landsat_multi/Single_band/', '*tif')
       stacked, meta = raster.layerstack(input, 'output.tif')

    '''

    import rasterio
    from rasterio.plot import reshape_as_raster
    import numpy as np

    files2stack = []
    stacked_array = []
    nbands = len(input)

    for file in input:
        with rasterio.open(file) as src:
            data = src.read(1)
            meta = src.meta
            files2stack.append(data)
    
    stacked_array = np.stack(files2stack, axis=-1)
    stacked = reshape_as_raster(stacked_array)
    meta.update({'count': nbands,
                            'driver': 'GTiff',
                            'compress': 'lzw'})
    
    # write out result 
    if output is not None:
        writeRaster(stacked, output, meta)
    else:
        return stacked, meta

# =========================================================================================== #
#               Merge  geotif files in a list using GDAL and VRT
# =========================================================================================== #
def mergeVRT(input: AnyStr, output: AnyStr, compress: bool=True, silent=True):
    '''
    Merge multiple geotif files using gdal VRT for better performance speed

    Parameters:
        input_files: List of input geotif files
        output_file: path of output tif file

    Example:
       input_list = tools.list_files('./test/', '*tif')
       raster.mergeVRT(input_list, './test/output.tif')

    '''

    import os
    from osgeo import gdal
    #  Create a temp vrt file
    vrt_file = 'merged.vrt'

    if compress is True:
        gdal.BuildVRT(vrt_file, input, options=['COMPRESS=LZW'])
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
def merge(input: AnyStr, output: Optional[AnyStr]=None, compress: Optional[AnyStr]='lzw'):
    '''
    Merge multiple geotif file using Rasterio 

    Parameters:
        input_files: List of input geotif files
        output_file: path of output tif file

    Example:
       input_list = tools.listFiles('./test/', 'tif)
       raster.merge(input_list, './test/output.tif')

    '''
    import rasterio
    from rasterio import merge 

    src_files = []
    for file in input:
        ds = rasterio.open(file)
        src_files.append(ds)

    fun_sum = merge.copy_sum
    fun_count = merge.copy_count

    mosaic_sum, out_trans = merge.merge(src_files, method=fun_sum)
    mosaic_count, out_trans = merge.merge(src_files, method=fun_count)

    mosaic_avg = mosaic_sum / mosaic_count
    
    meta = src_files[0].meta.copy()
    meta.update({"driver": "GTiff",
                            "height": mosaic_avg.shape[1],
                            "width": mosaic_avg.shape[2],
                            "transform": out_trans,
                            "compress": compress})
    
    # write out result 
    if output is not None:
        with rasterio.open(output, "w", **meta) as dest:
            dest.write(mosaic_avg)
        print(f"Finished merge raster files, the output is at {output}")
    # keep local
    else:
        return mosaic_avg, meta      


# =========================================================================================== #
#              Crop raster using shapefile or another image
# =========================================================================================== #
def crop(input, reference, input_meta: Optional[Dict]=None, reference_meta: Optional[Dict]=None, invert=False, nodata=None, output: Optional[AnyStr]=None):
    '''
    Crop raster opened by rasterio by shapefile vector or another image

    Parameters:
        input: image or data array input that need to crop
        reference: region of interest, shapefile opened by geopandas or another image or data array to crop
        input_meta: Optional metadata when input is data array
        reference_meta: Optional metadata when reference is data array
        invert: bool, if True, pixels inside shapefile will be masked 
        output: Optional string, whether write out geotif file to local directory

    Example:
        img = rasterio.open('./landsat_multi/landsat_img_test.tif', 'r')
        roi = gpd.read_file('./roi/roi.shp')
        clipped, meta = raster.crop(img, roi)

        import earthpy.plot as ep
        ep.plot_rgb(clipped, stretch=True, rgb=(3,2,1))

    '''
    import rasterio
    from rasterio.transform import Affine
    from shapely.geometry import mapping
    from shapely.geometry import box
    import geopandas as gpd
    import numpy as np

    ### Define input image
    # input is raster
    if isinstance(input, rasterio.DatasetReader):
        input_image = input
    # input is array
    elif isinstance(input, np.ndarray):
        if input_meta is None:
            raise ValueError('It requires metadata of input')
        else:
            if not os.path.exists('./tmp/'):
                os.makedirs('./tmp/')
            writeRaster(input, './tmp/tmp.tif', input_meta)
            input_image = rast('./tmp/tmp.tif')
    # Other input
    else:
        raise ValueError('Input data is not supported')

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
    # Reference is array
    elif isinstance(reference, np.ndarray):
        if reference_meta is None:
            raise ValueError('It requires metadata of reference')
        else:
            minx, miny = reference_meta['transform'] * (0, 0)
            maxx, maxy = reference_meta['transform']* (reference_meta['width'], reference_meta['height'])
            # define box
            bbox = box(minx, miny, maxx, maxy)
            poly_bound = gpd.GeoDataFrame({'geometry': [bbox]}, crs= reference_meta['crs'])
    else:
        raise ValueError('Reference data is not supported')   
    
    ### Define nodata
    if nodata is None:
        dataType = input_image.meta['dtype']
        if dataType.lower() == 'int8':
            nodata_value = 127
        elif dataType.lower() == 'uint8':
            nodata_value = 255
        elif dataType.lower() == 'int16':
            nodata_value = 32767
        elif dataType.lower() == 'uint16':
            nodata_value = 65535
        elif dataType.lower() == 'int32':
            nodata_value = 2147483647
        elif dataType.lower() == 'uint32':
            nodata_value == 4294967295
        elif dataType.lower() == 'float16':
            nodata_value = 65500
        elif dataType.lower() == 'float32':
            nodata_value == 999999
        else:
            nodata_value == 0
    else:
        nodata_value = nodata    
    
    ### Invert crop
    if invert is True:
        clipped, geotranform = rasterio.mask.mask(dataset=input_image, shapes= poly_bound.geometry.apply(mapping), crop=True, invert=True, nodata=-999999)
    else:
        clipped, geotranform = rasterio.mask.mask(dataset=input_image, shapes= poly_bound.geometry.apply(mapping), crop=True, nodata=-999999)

    # Define metadata
    meta  = input_image.meta
    meta.update({
        'height': clipped.shape[1],
        'width': clipped.shape[2],
        'transform': geotranform,
        'dtype': np.float32,
        'nodata': nodata_value})
    
    # Write output
    if output is not None:
        writeRaster(clipped, output, meta)
    else:
        return clipped, meta

# =========================================================================================== #
#              Mask raster using shapefile or another image
# =========================================================================================== #
def mask(input, reference, input_meta: Optional[Dict]=None, reference_meta: Optional[Dict]=None, invert=False, nodata=0, output: Optional[AnyStr]=None):
    '''
    Mask raster opened by rasterio by shapefile vector or another image

    Parameters:
        input: image or data array input that need to crop
        reference: region of interest, shapefile opened by geopandas or another image or data array to crop
        input_meta: Optional metadata when input is data array
        reference_meta: Optional metadata when reference is data array
        invert: bool, if True, pixels inside shapefile will be masked 
        output: Optional string, whether write out geotif file to local directory

    Example:
        img = rasterio.open('./landsat_multi/landsat_img_test.tif', 'r')
        roi = gpd.read_file('./roi/roi.shp')
        masked = raster.mask(img, roi)

        import earthpy.plot as ep
        ep.plot_rgb(masked, stretch=True, rgb=(3,2,1))

    '''
    import rasterio
    import shapely
    from shapely.geometry import mapping
    import geopandas as gpd
    import numpy as np
    from geonate import raster

    ### Define input image
    # input is raster
    if isinstance(input, rasterio.DatasetReader):
        input_image = input
    # input is array
    elif isinstance(input, np.ndarray):
        if input_meta is None:
            raise ValueError('It requires metadata of input')
        else:
            if not os.path.exists('./tmp/'):
                os.makedirs('./tmp/')
            writeRaster(input, './tmp/tmp.tif', input_meta)
            input_image = rast('./tmp/tmp.tif')
    # Other input
    else:
        raise ValueError('Input data is not supported')
    
    ### Define boundary
    if (isinstance(reference, gpd.GeoDataFrame)):
        poly = reference
        transform_poly = reference.transform
        crs_poly = reference.crs
    else:
        if isinstance(reference, rasterio.DatasetReader):
            ds_reference = reference.read(1)
            transform_poly = reference.meta['transform']
            crs_poly = reference.meta['crs']
        elif isinstance(reference, np.ndarray):
            ds_reference = reference[1, : , : ]
            transform_poly = reference_meta['transform']
            crs_poly = reference_meta['crs']
        else:
            raise ValueError('Data is not supported')
        
        masked = np.where(np.isnan(ds_reference), np.nan, 1)
        masked_convert = masked.astype(np.float32)
        
        shp = rasterio.features.shapes(masked_convert, mask= ~np.isnan(masked_convert), transform= transform_poly)
        poly = []
        values = []

        for shape, value in shp:
            if value == 1:
                poly.append(shapely.geometry.shape(shape))
                values.append(value)

        poly = gpd.GeoDataFrame({'geometry': poly, 'value': values})
        poly.set_crs(crs_poly.to_string(), inplace=True)
    
    ### Define nodata
    if nodata is None:
        dataType = input_image.meta['dtype']
        if dataType.lower() == 'int8':
            nodata_value = 127
        elif dataType.lower() == 'uint8':
            nodata_value = 255
        elif dataType.lower() == 'int16':
            nodata_value = 32767
        elif dataType.lower() == 'uint16':
            nodata_value = 65535
        elif dataType.lower() == 'int32':
            nodata_value = 2147483647
        elif dataType.lower() == 'uint32':
            nodata_value == 4294967295
        elif dataType.lower() == 'float16':
            nodata_value = 65500
        elif dataType.lower() == 'float32':
            nodata_value == 999999
        else:
            nodata_value == 0
    else:
        nodata_value = nodata        
    
    ### Invert mask
    if invert is True:
        masked_img, geotranform = rasterio.mask.mask(dataset=input_image, shapes= poly.geometry.apply(mapping), crop=True, invert=True, nodata=nodata_value)
    else:
        masked_img, geotranform = rasterio.mask.mask(dataset=input_image, shapes= poly.geometry.apply(mapping), crop=True, nodata= nodata_value)

    meta  = input_image.meta
    meta.update({
        'height': masked_img.shape[1],
        'width': masked_img.shape[2],
        'transform': geotranform,
        'dtype': np.float32,
        'nodata': nodata_value})
    
    # Write output
    if output is not None:
        raster.writeRaster(masked_img, output, meta)
    else:
        return masked_img, meta
    
    
# =========================================================================================== #
#              Preprojection raster image 
# =========================================================================================== #
def project(input, reference:Optional[AnyStr]=None, method: Optional[AnyStr]='near', input_meta: Optional[Dict]=None, res: Optional[float]=None, output: Optional[AnyStr]=None):
    '''
    Reproject raster data to a new projection and resample (if required)

    Parameters:
        input: Input rasterio image or data array
        reference: reference data of even another image or 'crs' string, e.g., 'EPSG:4326'
        method: resampling method (if changing resolution), optional, default method is 'nearest neighbor', other methods are at rasterio.Resampling. (e.g., nearest, cubic, bilinear, average)
        input_meta: Dict, required when input is data array
        res: Optional number, Resolution of the output in degree or meters depends on the output crs
        output: Optional string, whether write out geotif file to local directory
    
    Example:
        img = raster.rast('./Sample_data/landsat_multi/landsat_stacked.tif')
        ds = img.read()
        meta = img.meta
        arr, metadata = raster.project(input=ds, reference='EPSG:32648', input_meta=meta, res=1000)
       
    '''
    import rasterio
    from rasterio import warp
    import numpy as np

    ### Define input image
    # input is raster
    if isinstance(input, rasterio.DatasetReader):
        input_image = input.read()
        meta = input.meta
        # define boundary of input image
        left, bottom, right, top = input.bounds
    # input is array
    elif isinstance(input, np.ndarray):
        if input_meta is None:
            raise ValueError('It requires metadata of input')
        else:
            input_image = input
            meta = input_meta
            left, bottom = meta['transform'] * (0, 0)
            # define boundary of input image
            right, top = meta['transform']* (meta['width'], meta['height'])
    # Other input
    else:
        raise ValueError('Input data is not supported')

    # If reference is not given, it will take the CRS from input
    # the function now used for resampling
    if reference is None:
        dst_crs = meta['crs']
        if res is None:
            xsize, ysize = xsize, ysize = meta['transform'][0], meta['transform'][0]
        else:
            xsize, ysize = res, res
        transform, width, height = warp.calculate_default_transform(src_crs=meta['crs'], dst_crs=dst_crs, height=meta['height'], width=meta['width'], resolution=(xsize, ysize), left=left, bottom=bottom, right=right, top=top)

    else:
        # string of EPSG
        if isinstance(reference, str):
            dst_crs = reference
            if res is None:
                raise ValueError('Please provide output resolution')
            else:
                xsize, ysize = res, res
            transform, width, height = warp.calculate_default_transform(src_crs=meta['crs'], dst_crs=dst_crs, height=meta['height'], width=meta['width'], resolution=(xsize, ysize), left=left, bottom=bottom, right=right, top=top)

        # Take all paras from reference image
        elif isinstance(reference, rasterio.DatasetReader):
            dst_crs = reference.crs
            if res is None:
                xsize, ysize = reference.res
            else:
                xsize, ysize = res, res
            transform, width, height = warp.calculate_default_transform(src_crs=meta['crs'], dst_crs=dst_crs, height=meta['height'], width=meta['width'], resolution=(xsize, ysize), left=left, bottom=bottom, right=right, top=top)
        else:
            raise ValueError('Please define correct reference, it is CRS string or an image')
              
    # Update metadata
    meta_update = meta.copy()
    meta_update.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height,
    })

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

    # Running project 
    projected = np.empty((input_image.shape[0], height, width), dtype= meta['dtype'])
    for band in range(0, input_image.shape[0]):
        ds = input_image[band, : , : ]
        warp.reproject(source=ds, destination=projected[(band), :, :], src_transform= meta['transform'], dst_transform=transform, src_crs=meta['crs'], dst_crs=dst_crs, resampling= resampleAlg)
    
    # Write output
    if output is not None:
        writeRaster(projected, output, meta_update)
    else:
        return projected, meta_update

# =========================================================================================== #
#              Matching two images to have the same boundary
# =========================================================================================== #
def resample(input, factor, resample: AnyStr, method='near', meta: Optional[Dict]=None, output: Optional[AnyStr]=None):
    import rasterio
    from rasterio import warp
    import numpy as np

    ### Define input image
    # input is raster
    if isinstance(input, rasterio.DatasetReader):
        dataset = input.read()
        meta = input.meta
        left, bottom, right, top = input.bounds
        nbands = input.count
    # input is array
    elif isinstance(input, np.ndarray):
        if meta is None:
            raise ValueError('It requires metadata of input')
        else:
            dataset = input
            meta = meta
            left, bottom, right, top = getBounds(dataset, meta)
            if len(dataset.shape) > 2:
                nbands = dataset.shape[0]
            else:
                nbands = 1
    # Other input
    else:
        raise ValueError('Input data is not supported')
    
    #### Calculate new rows and columns
    if (resample.lower() == 'aggregate') or (resample.lower() == 'agg') or (resample.lower() == 'a'):
        new_height = meta['height'] // factor
        new_width = meta['width'] // factor
    elif (resample.lower() == 'disaggregate') or (resample.lower() == 'disagg') or (resample.lower() == 'd'):
        new_height = meta['height'] * factor
        new_width = meta['width'] * factor
    else:
        raise ValueError('Resample method is not supported ["aggregate", "disaggregate"]')

    transform_new, width, height = warp.calculate_default_transform(src_crs=meta['crs'], dst_crs=meta['crs'], width=new_width, height=new_height, left=left, bottom=bottom, right=right, top=top)

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

    # Define the metadata for the destination raster
    metadata = meta.copy()
    metadata.update({
        'transform': transform_new,
        'width': new_width,
        'height': new_height, 
        'dtype': np.float32
    })

    resampled = np.empty((nbands, new_height, new_width), dtype=np.float32)
    for band in range(0, nbands):
        if nbands <= 1:
            ds = dataset
        else:
            ds = dataset[band, : , : ]
        warp.reproject(source=ds, destination=resampled[band, :, :], src_transform= meta['transform'], dst_transform= transform_new, src_crs=meta['crs'], dst_crs=input.crs, resampling= resampleAlg)

    # Write output
    if output is not None:
        writeRaster(resampled, output, metadata)
    else:
        return resampled, metadata
  

# =========================================================================================== #
#              Matching two images to have the same boundary
# =========================================================================================== #
    
def match(input, reference, method: AnyStr='near', input_meta: Optional[Dict]=None, reference_meta: Optional[Dict]=None):
    '''
    Match input image to the reference image in terms of projection, resolution, and bound extent
    It returns image within the bigger boundary
    
    Parameters:
        input: rasterio objective needs to match 
        reference: rasterio object taken as reference to match the input image
        method: optional string defines resampling method (if applicable) to resample if having different resolution
        input_meta: Optional Dict, metadata of input required when input is data array
        reference_meta: Optional Dict, metadata of reference required when reference is data array
        
    Example:
        img1 = rasterio.open('./Sample_data/landsat_multi/scene/landsat_img_01.tif')
        img2 = rasterio.open('./Sample_data/landsat_multi/scene/landsat_img_02.tif')
    
        img1_matched, meta = raster.match(img1, img2)
        ep.plot_bands(img1_matched)
    
    '''
    import rasterio
    from rasterio import warp
    from rasterio.transform import from_bounds
    import numpy as np
    
    ### Define input image
    # input is raster
    if isinstance(input, rasterio.DatasetReader):
        input_image = input.read()
        meta = input.meta
    # input is array
    elif isinstance(input, np.ndarray):
        if input_meta is None:
            raise ValueError('It requires metadata of input')
        else:
            input_image = input
            meta = input_meta
    # Other input
    else:
        raise ValueError('Input data is not supported')
    
    ### Define reference image
    if isinstance(reference, rasterio.DatasetReader):
        reference_image = reference.read()
        meta_reference = reference.meta
    # input is array
    elif isinstance(reference, np.ndarray):
        if reference_meta is None:
            raise ValueError('It requires metadata of reference')
        else:
            reference_image = reference
            meta_reference = reference_meta
    # Other input
    else:
        raise ValueError('Input data is not supported')
    
    ### Check some conditions
    if meta["crs"] != meta_reference['crs']:
        print('Input and reference images have different Projection')
        print('Output projection will take the reference projection')
    if meta['transform'][0] != meta_reference['transform'][0]:
        print('Input and reference images have different resolution')
        print('Ouput resolution will take the reference resolution')     
    
    # get general extent from two images
    ext_input = getBounds(input_image, meta)
    ext_reference = getBounds(reference_image, meta_reference)
    
    ext = ext_input
    ext = (
        min(ext[0], ext_reference[0]),
        min(ext[1], ext_reference[1]),
        max(ext[2], ext_reference[2]),
        max(ext[3], ext_reference[3])
        )
        
    # calculate new height and width
    resolution = meta_reference['transform'][0]    
    width_new = int((ext[2]  - ext[0]) / resolution)
    height_new = int((ext[3] - ext[1]) / resolution)
    
    # calculate new transform
    transform_new = from_bounds(ext[0], ext[1], ext[2], ext[3], width_new, height_new)
        
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
    
    # Reproject to match
    if len(input_image.shape) > 2:
        nbands = input_image.shape[0]
    else:
        nbands = 1

    matched = np.empty((nbands, height_new, width_new), dtype=np.float32)
    for band in range(0, nbands):
        if nbands <= 1:
            ds = input_image
        else:
            ds = input_image[band, : , : ]
        warp.reproject(source=ds, destination=matched[band, :, :], src_transform= meta['transform'], dst_transform= transform_new, src_crs=meta['crs'], dst_crs=meta_reference['crs'], resampling= resampleAlg)
    
    # match out other values
    match_masked = np.where(matched == 0, np.nan, matched)
    match_masked = match_masked.astype(np.float32)

    # update metadata
    meta_update = meta.copy()
    meta_update.update({
        'crs': meta_reference['crs'],
        'transform': transform_new,
        'width': width_new,
        'height': height_new,
        'dtype': np.float32
    })
    
    return match_masked, meta_update
   

# =========================================================================================== #
#              Calculate normalized difference index 
# =========================================================================================== #
def normalizedDifference(input, band1, band2, meta: Optional[Dict]=None, output: Optional[AnyStr]=None):
    '''
    Calculate normalized difference index

    Parameters:
        input: rasterio object or data array, input with multiple bands 
        band1: numeric, order of the first band
        band2: numeric, order of the second band
        meta: optional dict, metadata in case input is data array
        output: Optional string, whether write out geotif file to local directory
    
    Example:
        img = rasterio.open('./Sample_data/landsat_multi/landsat_stacked.tif')
        ndvi = raster.normalizedDifference(img, 4, 3)
       
    '''    
    import numpy as np
    import rasterio

    # input is raster
    if isinstance(input, rasterio.DatasetReader):
        dataset = input.read()
        meta = input.meta
    # input is array
    elif isinstance(input, np.ndarray):
        if meta is None:
            raise ValueError('It requires metadata')
        else:
            dataset = input
            meta = meta
    # Other input
    else:
        raise ValueError('Input data is not supported')

    # Extract data
    ds_band1 = dataset[band1+1, : , : ]
    ds_band2 = dataset[band2+1, : , : ]

    # calculate index
    normalized_index  = (ds_band1.astype(float) - ds_band2.astype(float)) / (ds_band1 + ds_band2)
    normalized_index = normalized_index.astype(np.float32)
    # remove outliers
    normalized_index[(normalized_index < -1) | (normalized_index > 1)] = np.nan

    meta.update({'dtype': np.float32})

    # Write output
    if output is not None:
        writeRaster(normalized_index, output, meta)
    else:
        return normalized_index, meta



# =========================================================================================== #
#              Reclassify image
# =========================================================================================== #
def reclassify(input, breakpoints, classes, meta: Optional[Dict]=None, output: Optional[AnyStr]=None):
    '''
    Reclassify image with discrete or continuous values

    Parameters:
        input: raster or data array input
        breakpoints: number list, defines a breakpoint value for reclassifcation, e.g., [ -1, 0, 1]
        classes: number list, define classes, number of classes equal number of breakpoints minus 1
        meta: meta: optional dict, metadata in case input is data array
        output: Optional string, whether write out geotif file to local directory

    Example:
        img = rasterio.open('./Sample_data/landsat_multi/landsat_stacked.tif')
        rc = raster.reclassify(img, breakpoints=[float('-inf'), -0.5, 0, 0.5, float('-inf')], classes=[1, 2, 3, 4])
       
    '''    
    import rasterio
    import numpy as np

    ### Check input data
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
        if meta is None:
            raise ValueError('It requires metadata')
        else:
            if (len(input.shape)) > 2 and (input.shape[0] > 1):
                raise ValueError('Input data has more than one band')
            else:
                dataset = input
                meta = meta
    # Other input
    else:
        raise ValueError('Input data is not supported')

    ####
    # Create unique values and empty data array to store reclassified result 
    uniques = np.unique(dataset)
    reclassified = np.zeros_like(dataset)
        
    ####
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
    
    # Write output
    if output is not None:
        writeRaster(reclassified, output, meta)
    else:
        return reclassified, meta
    

# =========================================================================================== #
#              Extract raster values of all bands and create dataframe
# =========================================================================================== #
def values(input, meta: Optional[AnyStr]=None, na_rm: Optional[bool]=True, names: Optional[list]=None, prefix: Optional[AnyStr]=None):
    '''
    Extract all pixel values of image and create dataframe from them, each band is a column

    Parameters:
        input: rasterio image or data array
        meta: meta: optional dict, metadata in case input is data array
        names: optional list, given expected names
        prefix: optional string, given character before each band name
    
    Example:
        img = rasterio.open('./Sample_data/landsat_multi/landsat_crop.tif')
        df = raster.values(img, prefix='Band')
        df.head(5)
       
    '''    
    import rasterio
    import numpy as np
    import pandas as pd

    ### Check input data
    # input is raster
    if isinstance(input, rasterio.DatasetReader):
        dataset = input.read()
        meta = input.meta
    # input is array
    elif isinstance(input, np.ndarray):
        if meta is None:
            raise ValueError('It requires metadata')
        else:
            dataset = input
            meta = meta
    # Other input
    else:
        raise ValueError('Input data is not supported')
    
    ########
    nbands = dataset.shape[0]
    bands_array = [dataset[band, : , : ].flatten() for band in range(0, nbands)]
    
    if names is not None:
        if len(names) != nbands:
            raise ValueError('Length of name should be equal to number of bands')
        else:
            if prefix is None:
                data = pd.DataFrame(np.array(bands_array).T, columns=names)
            else:
                names_new = [f'{prefix}{name}' for name in names]
                data = pd.DataFrame(np.array(bands_array).T, columns=names_new)
    else:
        if prefix is None:
            data = pd.DataFrame(np.array(bands_array).T, columns=[f'B{i}' for i in range(1,nbands +1)])
        else:
            data = pd.DataFrame(np.array(bands_array).T, columns=[f'{prefix}{i}' for i in range(1, nbands +1)])
    
    ####### 
    # Remove NA values or not
    if na_rm is True: 
        data_out = data.dropna().reset_index(drop=True)
    else:
        data_out = data

    return data_out


 # =========================================================================================== #
#              Extract pixel values at GCP (points or polygon)
# =========================================================================================== #       
def extractValues(input: AnyStr, roi: AnyStr, field: AnyStr=None, meta: Optional[AnyStr]=None, dataframe: Optional[bool]=True, names: Optional[list]=None, na_rm: bool=True, nodata=None, prefix: Optional[AnyStr]=None, tail=True):
    '''
    Extract pixel values in GCP for both point and polygon 

    Parameters:
        input: rasterio image or data array
        roi: variable indicates name of shapefile where GCP points located, read by geopandas 
        field: string, but the value of field must be number, the field name in shapefile GCP to extract label values, e.g., 'class'
        meta: optional dict, metadata in case input is data array
        dataframe: optional bool, whether to return dataframe or separate X, y arrays
        names: optional list, given expected names
        na_rm: bool, remove NA value from the output or not
        prefix: optional string, given character before each band name

    Example:
       img = rasterio.open('./Sample_data/landsat_multi/landsat_stacked.tif')
       roi = gpd.read_file('./Sample_data/shapefile/GCP_polys.shp')
       
       # return combined dataframe
       df = raster.extractValues(img, roi, field='class')
       df.head(5)
       
       # return separate array of X and y
       X, y = raster.extractValues(img, roi, field='class', dataframe=False)

    '''
    import rasterio
    from rasterio.plot import reshape_as_image
    import numpy as np
    from shapely.geometry import mapping
    from rasterio import mask
    import pandas as pd

    if field is None:
        raise ValueError('Please provide field name')
    else:
        ### Define input image
        # input is raster
        if isinstance(input, rasterio.DatasetReader):
            input_image = input
        # input is array
        elif isinstance(input, np.ndarray):
            if meta is None:
                raise ValueError('It requires metadata of input')
            else:
                if not os.path.exists('./tmp/'):
                    os.makedirs('./tmp/')
                writeRaster(input, './tmp/tmp.tif', meta)
                input_image = rast('./tmp/tmp.tif')
        # Other input
        else:
            raise ValueError('Input data is not supported')
        
        ########

        ### Define nodata
        if nodata is None:
            dataType = input_image.meta['dtype']
            if dataType.lower() == 'int8':
                nodata_value = 127
            elif dataType.lower() == 'uint8':
                nodata_value = 255
            elif dataType.lower() == 'int16':
                nodata_value = 32767
            elif dataType.lower() == 'uint16':
                nodata_value = 65535
            elif dataType.lower() == 'int32':
                nodata_value = 2147483647
            elif dataType.lower() == 'uint32':
                nodata_value == 4294967295
            elif dataType.lower() == 'float16':
                nodata_value = 65500
            elif dataType.lower() == 'float32':
                nodata_value == 999999
            else:
                nodata_value == 0
        else:
            nodata_value = nodata   

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
            cropped, transform = mask.mask(input_image, poly, crop=True, nodata=nodata_value)

            # Reshape dataset in form of (values, bands)
            cropped_reshape = reshape_as_image(cropped)
            reshapped = cropped_reshape.reshape(-1, nbands)

            # Append 1D array y
            y = np.append(y, [roi[field][index]] * reshapped.shape[0])
            
            # vertical stack 2D array X
            X = np.vstack((X, reshapped))
        
        # Remove NAN value or not
        if na_rm is True: 
            data = np.hstack((X, y.reshape(y.shape[0], 1)))
            data_na = data[~np.isnan(data).any(axis=1)]
            data_nodata = data_na[~(data_na == nodata_value).any(axis=1)]

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
        
        # Do not remove NAN values
        else: 
            # return dataframe
            if dataframe is True:
                y_reshape = y.reshape(-1,1)
                arr = np.hstack([X, y_reshape])

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

            else:
                return X, y
    

# =========================================================================================== #
#              Normalize raster data
# =========================================================================================== #      
def normalized(input, meta: Optional[AnyStr]=None, output: Optional[AnyStr]=None):
    '''
    Normalize raster data to rearrange raster values from 0 to 1

    Parameters:
        input: rasterio image or data array
        meta: optional dict, metadata in case input is data array
        output: Optional string, whether write out geotif file to local directory
    
    Example:
       img = rasterio.open('./Sample_data/landsat_multi/landsat_stacked.tif')
       normalized, meta = raster.normalized(img)       

    '''
    import rasterio
    import numpy as np
    import pandas as pd

    ### Check input data
    if isinstance(input, rasterio.DatasetReader):
        dataset = input.read()
        meta = input.meta
    elif isinstance(input, np.ndarray):
        if meta is None:
            raise ValueError('Please provide input metadata')
        else:
            dataset = input
            meta = meta
    else:
        raise ValueError('Input data is not supported')
    
    ### Find max min values
    maxValue = np.nanmax(dataset)
    minValue = np.nanmin(dataset)

    ### Create empty data array to store output
    normalized = np.zeros_like(dataset, dtype=np.float32)

    ### Run normalization
    for i in range(0, dataset.shape[0]):
        band = dataset[i, : , : ]
        band_norm = (band.astype(float)  - minValue) / (maxValue  - minValue)
        normalized[i, : , : ] = band_norm
        band_norm = None

    ### update meta
    meta.update({'dtype': np.float32})

    ### return result 
    # Write output
    if output is not None:
        writeRaster(normalized, output, meta)
    else:
        return normalized, meta
    

# =========================================================================================== #
#              Return min and max values of array or raster
# =========================================================================================== #
def mimax(input, digit=3):
    '''
    Calculate maximum and minimum values of raster or array

    Parameters:
        input: rasterio image or data array
        digit: Optional string, whether write out geotif file to local directory
    
    Example:
       img = rasterio.open('./Sample_data/landsat_multi/landsat_stacked.tif')
       minvalue, maxvalue = raster.mimax(img)

    '''
    
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
    '''
    Calculate pixel size (area), the input has to be in the projection of 'EPSG:4326'. If not, it can be reprojected by "project" function

    Parameters:
        input: rasterio image or data array
        unit: string, default is "km", the unit to calculate area
        meta: optional dict, metadata in case input is data array
        output: Optional string, whether write out geotif file to local directory
    
    Example:
       img = raster.rast('./Sample_data/temperature.tif')
       cellArea, meta = raster.cellSize(img, unit='km)    

    '''
    import rasterio
    import numpy as np
    import pandas as pd

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
        writeRaster(outArea, output, meta)
    else:
        return outArea, meta
