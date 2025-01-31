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
def plot(input, cmap='Greys_r', **kwargs):
    """Plot a raster image or data array using earthpy.

    Args:
        input (DatasetReader | np.ndarray): Rasterio image or data array
        cmap (str, optional): Colormap for the plot. Defaults to 'Greys_r'.
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
    ep.plot_bands(dataset, cmap=cmap, **kwargs)


# =========================================================================================== #
#               RGB composite plot
# =========================================================================================== #
def plotRGB(input, rgb=(0, 1, 2), stretch=True, **kwargs):
    """
    Plot a 3-band RGB image using earthpy.

    Args:
        input (rasterio.DatasetReader | np.ndarray): Rasterio image or data array.
        rgb (tuple, optional): Indices of the RGB bands. Defaults to (0, 1, 2).
        stretch (bool, optional): Apply contrast stretching. Defaults to True.
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
    ep.plot_rgb(dataset, rgb= rgb, stretch=stretch, **kwargs)


    
