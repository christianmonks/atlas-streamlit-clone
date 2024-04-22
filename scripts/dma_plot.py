import json
import pandas as pd
import numpy as np
import plotly.graph_objs as go

__all__ = ['get_choropleths']

def get_centroids(
        dma_geo:dict
):
    """
    Calculate centroids of the given DMA (Designated Market Area) geometry.

    Parameters:
    dma_geo (dict): A dictionary containing the geometry information of the DMA.

    Returns:
    list: A list of tuples containing the latitude and longitude coordinates of the centroids.

    Raises:
    Exception: If the geometry type is not supported.

    Note:
    This function assumes that the input DMA geometry is in GeoJSON format.
    """
    geo_type = dma_geo.get('features')[0].get('geometry').get('type')
    geo_coords = dma_geo.get('features')[0].get('geometry').get('coordinates')

    centroids = []
    if geo_type == 'Polygon':
        centroid = polylabel.polylabel(Polygon(geo_coords[0]))
        centroids.append((centroid.y,centroid.x))
    elif geo_type == 'MultiPolygon':
        for polygon in geo_coords:
            centroid = polylabel.polylabel(Polygon(polygon[0]))
            centroids.append((centroid.y,centroid.x))
    else:
        raise Exception(f'Unsupported geometry type: {geo_type}')

    return centroids

def get_perf_color(val, perf_df):
    """
    Determine the color based on the performance value.

    This function calculates the color based on the given performance value.
    It defines color scales for different performance levels and assigns colors accordingly.

    Parameters:
    val (float): The performance value to determine the color for.

    Returns:
    str: The color code corresponding to the performance level of the input value.

    Note:
    - The performance value should be between 0 and 1.
    - Color scales are predefined within the function.
    """

    # Define color scales here
    high_color = '#17800b'
    normal_color = '#2fff13'
    caution_color = '#ff8521'
    danger_color = '#ff0a2b'
    default_color = 'grey'

    high = perf_df['Score'].mean() + (perf_df['Score'].std())
    normal = perf_df['Score'].mean()
    caution =perf_df['Score'].mean() - (perf_df['Score'].std())

    # Only color sizable DMA's defined by store_tolerance
    if val > high:
        return high_color
    elif val > normal:
        return normal_color
    elif val > caution:
        return caution_color
    elif (val > 0) & (val <= caution):
        return danger_color
    else:
        return default_color


def get_dma_info(dma, perf_df, metric):
    """
    Get information about a DMA (Designated Market Area) including its code, name, coordinates, and performance.

    This function extracts information about a DMA from the provided DMA GeoJSON object and a DataFrame containing performance data. It retrieves the DMA code, name, latitude, and longitude coordinates. Additionally, it calculates the performance value for the DMA based on the specified metric and determines the color corresponding to its performance level.

    Parameters:
    dma (dict): A dictionary containing the DMA information in GeoJSON format.
    perf_df (DataFrame): A DataFrame containing performance data, including DMA codes and performance metrics.
    metric (str): The performance metric to use for determining the DMA's performance.

    Returns:
    dict: A dictionary containing the DMA information including its code, name, latitude and longitude coordinates, performance color, and performance value.

    Note:
    - The 'metric' parameter should be a column name present in the 'perf_df' DataFrame.
    - This function relies on the 'get_perf_color' function to determine the color based on performance.
    """

    feature = dma.get('features')[0]

    # Extract DMA information
    dma_code = int(feature.get('id'))
    dma_name = feature.get('properties').get('dma1')

    lat = feature.get('properties').get('latitude')
    lon = feature.get('properties').get('longitude')

    perf_val = 0

    dma_s = perf_df.query(f'dma_code == {dma_code}').reset_index(drop=True)

    if dma_s.shape[0] == 0:
        perf_color = get_perf_color(0, perf_df)
    else:
        perf_val = dma_s.at[0, metric]
        perf_color = get_perf_color(perf_val, perf_df)

    return {
        'dma_code': dma_code,
        'dma_name': dma_name,
        'lat_ctr': lat,
        'lon_ctr': lon,
        'perf_color': perf_color,
        'perf_val': perf_val
    }


def get_choropleths(dmas_geo, perf_df, metric):
    """
    Generate choropleth data for DMAs (Designated Market Areas) based on performance metrics.

    This function generates choropleth data for DMAs using the provided DMA GeoJSON object and a DataFrame containing performance data. It calculates the performance information for each DMA, creates choropleth traces based on the DMA geometries, and adds markers on the centroids of each DMA for enabling hoverboxes.

    Parameters:
    dmas_geo (list): A list of dictionaries containing DMA information in GeoJSON format.
    perf_df (DataFrame): A DataFrame containing performance data, including DMA codes and performance metrics.
    metric (str): The performance metric to use for determining DMA performance.

    Returns:
    list: A list of plotly trace objects representing choropleth data.

    Note:
    - The 'metric' parameter should be a column name present in the 'perf_df' DataFrame.
    - This function relies on the 'get_dma_info' function to retrieve DMA information.
    - The 'create_area' function is assumed to create choropleth area traces.
    """

    data = []
    lat_centers = []
    lon_centers = []
    center_tooltips = []

    for dma in dmas_geo:

        dma_info = get_dma_info(dma, perf_df, metric)

        # Extract all polygons in the DMA
        geo_type = dma.get('features')[0].get('geometry').get('type')
        geo_coords = dma.get('features')[0].get('geometry').get('coordinates')
        dma_traces = []

        if geo_type == 'Polygon':
            x_coords, y_coords = zip(*geo_coords[0])

            data.append(create_area(x_coords, y_coords, dma_info.get('perf_color')))
        elif geo_type == 'MultiPolygon':

            for polygon in geo_coords:
                x_coords, y_coords = zip(*polygon[0])

                dma_traces.append(create_area(x_coords, y_coords, dma_info.get('perf_color')))
        else:
            raise Exception(f'Unsupported geometry type: {geo_type}')

        data.extend(dma_traces)

        # Add markers on centroid for this DMA to enable hoverboxes within each DMA region
        lat_centers.append(dma_info.get('lat_ctr'))
        lon_centers.append(dma_info.get('lon_ctr'))
        center_tooltips.append('DMA: {:s} Market Score: {:.3f}'.format(dma_info.get('dma_name'),
                                                                         dma_info.get('perf_val')))

    # Define the centers object for plotly to plot all these hidden markers to enable tooltips
    centers = dict(
        type='scatter',
        mode='markers',
        showlegend=False,
        marker=dict(size=5, opacity=0),
        text=center_tooltips,
        x=lon_centers,
        y=lat_centers,
        hoverinfo='text',
        hoverlabel=dict(bgcolor='white')
    )

    data.append(centers)

    return data

def create_area(x_coords, y_coords, perf_color):
    """
    Create a plotly Scatter trace object to define a filled area.

    This function creates a plotly Scatter trace object that defines a filled area using the provided x and y coordinates.
    The area is filled with the specified color.

    Parameters:
    x_coords (list): List of x coordinates defining the boundary of the area.
    y_coords (list): List of y coordinates defining the boundary of the area.
    perf_color (str): The color to fill the area.

    Returns:
    dict: A plotly Scatter trace definition for the filled area.

    """
    return {
        'type': 'scatter',
        'showlegend': False,
        'mode': 'lines',
        'line': {'color': 'black', 'width': 1},
        'x': x_coords,
        'y': y_coords,
        'fill': 'toself',
        'fillcolor': perf_color,
        'hoverinfo': 'none'
    }

