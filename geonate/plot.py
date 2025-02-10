"""
The visualization module

"""
# import common packages 

from typing import AnyStr, Dict, Optional

##############################################################################################
#                                                                                                                                                                                                          #
#                       Main functions                                                                                                                                                         #
#                                                                                                                                                                                                           #
##############################################################################################

# =========================================================================================== #
#               Display all available colormaps in Matplotlib
# =========================================================================================== #
def colormaps():   
    """
    Display all available colormaps in Matplotlib.
    
    This function generates a plot that shows all the colormaps available in Matplotlib.
    Each colormap is displayed as a horizontal gradient bar.

    """ 
    import numpy as np
    import matplotlib.pyplot as plt

    # Get all colormaps available in Matplotlib
    colormaps = plt.colormaps()

    # Generate a gradient to display colormaps
    gradient = np.linspace(0, 1, 256).reshape(1, -1)

    # Set figure size
    fig, ax = plt.subplots(figsize=(10, len(colormaps) * 0.25))

    # Loop through colormaps and display them
    for i, cmap in enumerate(colormaps):
        ax.imshow(np.vstack([gradient] * 5), aspect='auto', cmap=cmap, extent=[0, 10, i, i + 1])

    # Formatting
    ax.set_yticks(np.arange(len(colormaps)) + 0.5)
    ax.set_yticklabels(colormaps)
    ax.set_xticks([])
    ax.set_title("Matplotlib Colormaps", fontsize=12, fontweight="bold")
    ax.set_ylim(0, len(colormaps))

    plt.show()


# =========================================================================================== #
#               Simple plot band           
# =========================================================================================== #
def plot_bands(input, cmap='Greys_r', cols=3, figsize=(10,10), cbar=True, **kwargs):
    """Plot a raster image or data array using earthpy.

    Args:
        input (DatasetReader | np.ndarray): Rasterio image or data array
        cmap (str, optional): Colormap for the plot. Defaults to 'Greys_r'.
        cols (int): Numbers of column on the plot. Defaults to cols = 3.
        figsize (numeric tuple): Width and Height. Defaults to (10, 10) inches.
        cbar (bool): Show color cbar. Defaults to True.  
        **kwargs (AnyStr, optional): All optional parameters taken from earthpy.plot.plot_bands(), such as cmap='Spectral' for color shade

    """
    import numpy as np
    import rasterio
    import earthpy.plot as ep

    ### Check input data
    if isinstance(input, rasterio.DatasetReader):
        dataset = input.read()
    elif isinstance(input, np.ndarray):
        dataset = input
    else:
        raise ValueError('Input data is not supported')
    
    # Visualize the input dataset
    ep.plot_bands(dataset, cmap=cmap, cols=cols, figsize=figsize, cbar=cbar,**kwargs)


# =========================================================================================== #
#               RGB composite plot
# =========================================================================================== #
def plotRGB(input, rgb=(0, 1, 2), stretch=True, str_clip: int = 2, figsize=(10,10), **kwargs):
    """
    Plot a 3-band RGB image using earthpy.

    Args:
        input (rasterio.DatasetReader | np.ndarray): Rasterio image or data array.
        rgb (tuple, optional): Indices of the RGB bands. Defaults to (0, 1, 2).
        stretch (bool, optional): Apply contrast stretching. Defaults to True.    
        str_clip (int): The percentage of clip to apply to the stretch. Default = 2 (2 and 98).  
        figsize (numeric tuple): Width and Height. Defaults to (10, 10) inches.
        **kwargs: Additional optional parameters for earthpy.plot.plot_rgb(), such as stretch=True for contrast stretching.

    """    
    import numpy as np
    import rasterio
    import earthpy.plot as ep

    ### Check input data
    if isinstance(input, rasterio.DatasetReader):
        dataset = input.read()
    elif isinstance(input, np.ndarray):
        dataset = input
    else:
        raise ValueError('Input data is not supported')
    
    # Check data dimension to make sure it is a multiple band image
    if len(dataset) <= 2:
        raise ValueError('Image has only one band, please provide at least 3-band image')
    
    # Visualize the input dataset
    ep.plot_rgb(dataset, rgb= rgb, stretch=stretch, str_clip=str_clip, figsize=figsize, **kwargs)


# =========================================================================================== #
#               Plot raster overlay with basemap
# =========================================================================================== #
def plot_raster(input, layername: Optional[AnyStr]=None, rgb: Optional[list]=None, stretch: Optional[AnyStr]='linear', brightness: Optional[float]=None, contrast: Optional[float]=None, opacity: Optional[float]=1, zoom: Optional[float]=5, basemap: Optional[AnyStr]='OSM', output: Optional[AnyStr]= None):
    """
    Plots a basemap with an overlay of raster data.

    Args:
        input (DatasetReader): The input raster dataset.
        layername (Anstr, optional): Layer name of image.
        rgb (list, optional): List of RGB bands to visualize. Defaults to None.
        stretch (AnyStr, optional): Stretch method for the image ('linear', 'hist', 'custom'). Defaults to 'linear'.
        brightness (float, optional): Brightness value for custom stretch. Defaults to None.
        contrast (float, optional): Contrast value for custom stretch. Defaults to None.
        opacity (float, optional): Opacity of the image overlay. Defaults to 1.
        zoom (float, optional): Initial zoom level of the map. Defaults to 5.
        basemap (AnyStr, optional): Basemap type ('OSM', 'CartoDB Positron', 'CartoDB Dark Matter', 'OpenTopoMap', 'Esri Satellite', 'Esri Street Map', 'Esri Topo', 'Esri Canvas'). Defaults to 'OSM'.
        output (AnyStr, optional): File path to write out html file to local directory. Defaults to None.

    Returns:
        folium.Map: A folium map object with the raster data overlay.

    """
    import folium
    from folium.raster_layers import ImageOverlay
    import rasterio
    import numpy as np
    from .common import meter2degree, get_extent_local
    from .processor import reproject
    

    ### Check input data is raster or not and extract information
    if isinstance(input, rasterio.DatasetReader):
        # Convert image to lat/long if input is not in lat/long system
        crs = input.crs.to_string()
        if crs == "EPSG:4326":
            input_converted = input
        else:
            resolution_degree = meter2degree(input.res[0])
            input_converted = reproject(input, reference='EPSG:4326', res=resolution_degree)

        # Extract data from image to visualize
        if (input.count <= 2):
            print('Input image/data has less than 2 bands, it will load the first band only')
            dataset = input.read(1)
            imgData = dataset[:, :, np.newaxis]
        elif (input.count >= 3):
            if rgb is None: 
                raise ValueError('Input is multiple band image, please provide rgb bands to visualize [3,2,1]')
            else:
                dataset = input.read(rgb)
                imgData = np.transpose(dataset, (1, 2, 0)) # Transpose from raster dims (bands, width, height) to image dims (width, height, bands)
    else:
        raise ValueError("Input data is not supported. It must be raster image")

    # Check stretch method
    if stretch is None:
        data = imgData

    elif stretch.lower() == 'linear':
        data = np.clip((imgData  - imgData.min()) / (imgData.max() - imgData.min()) * 255, 0, 255).astype(np.uint8) # linear stretching based on min max values

    elif stretch.lower() == 'hist' or stretch.lower() == 'histogram':
        from skimage import exposure
        data = exposure.equalize_hist(imgData)  # This returns a floating point image with values between 0 and 1
        data = (data * 255).astype(np.uint8)  # Convert back to 8-bit image for display

    elif stretch.lower() == 'custom':
        if (contrast is None) or (brightness is None):
            raise ValueError("contrast and brightness must be given for custom stretching method")
        else:
            data = np.clip((imgData * contrast + brightness), 0, 255).astype(np.uint8)

    else: 
        raise ValueError("Stretch method is not supported ('linear', 'hist', 'custom')")
    
    # Get Bounds values
    left, bottom, right, top = get_extent_local(input_converted)[0]
    lat_center = (top + bottom) / 2
    lon_center = (left + right)/ 2
    bounds = [[bottom, left], [top, right]]
    
    # Create overlay image
    if layername is not None:
        image_overlay = ImageOverlay(image= data, bounds= bounds, opacity= opacity, name=layername)
    else:
        image_overlay = ImageOverlay(image= data, bounds= bounds, opacity= opacity, name='Layer')

    # Add the image overlay to the map

    # Take basemap
    if basemap.lower() == 'openstreetmap' or basemap.lower() == 'osm' or basemap.lower() == 'open street map':
        basemap_name = 'OpenStreetMap'
        m = folium.Map(location=[lat_center, lon_center], zoom_start= zoom, tiles=basemap_name)

    elif basemap.lower() == 'cartodbpositron' or basemap.lower() == 'cartodb positron' or basemap.lower() == 'light' :
        basemap_name = 'Cartodb Positron'
        m = folium.Map(location=[lat_center, lon_center], zoom_start= zoom, tiles=basemap_name)

    elif basemap.lower() == 'cartodbdarkmatter' or basemap.lower() == 'cartodb dark matter' or basemap.lower() == 'cartodb dark' or basemap.lower() == 'dark':
        basemap_name = 'Cartodb dark_matter'
        m = folium.Map(location=[lat_center, lon_center], zoom_start= zoom, tiles=basemap_name)
    
    elif basemap.lower() == 'opentopomap' or basemap.lower() == 'opentopo' or basemap.lower() == 'topo':
        m = folium.Map(location=[lat_center, lon_center], zoom_start= zoom)
        folium.TileLayer(
            tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
            attr='&copy; Topo Map',
            name='Open Topo Map'
        ).add_to(m)

    elif basemap.lower() == 'esri satellite' or basemap.lower() == 'esrisatellite' or basemap.lower() == 'satellite':
        m = folium.Map(location=[lat_center, lon_center], zoom_start= zoom)
        folium.TileLayer(
            tiles= 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='&copy; Esri',
            name='Esri Satellite'
        ).add_to(m)

    elif basemap.lower() == 'esri street' or basemap.lower() == 'esristreet' or basemap.lower() == 'streetmap' or basemap.lower() == 'street map':
        m = folium.Map(location=[lat_center, lon_center], zoom_start= zoom)
        folium.TileLayer(
            tiles= 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
            attr='&copy; Esri',
            name='Esri Street Map'
        ).add_to(m)

    elif basemap.lower() == 'esri topo' or basemap.lower() == 'esritopo':
        m = folium.Map(location=[lat_center, lon_center], zoom_start= zoom)
        folium.TileLayer(
            tiles= 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
            attr='&copy; Esri',
            name='Esri Topo Map'
        ).add_to(m)

    elif basemap.lower() == 'esri canvas' or basemap.lower() == 'esricanvas' or basemap.lower() == 'canvas':
        m = folium.Map(location=[lat_center, lon_center], zoom_start= zoom)
        folium.TileLayer(
            tiles= 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
            attr='&copy; Esri',
            name='Esri Canvas Gray'
        ).add_to(m)

    else:
        raise ValueError("Basemap is not supported, please select one of these maps ('OSM', 'CartoDB Positron', 'CartoDB Dark Matter', 'OpenTopoMap', 'Esri Satellite', 'Esri Street Map', 'Esri Topo', 'Esri Canvas')")

    # Add image to basemap    
    image_overlay.add_to(m)
    folium.LayerControl().add_to(m)

    # Save map
    if output is not None:
        m.save(output)
    else:
        pass

    return m  


    
