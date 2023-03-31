import jax.numpy as jnp
import numpy as np
from jax import vmap

def adstocked_advertising_jnp(x, adstock_rate, L):
    """
    Apply adstock transformation to the input array.

    The function computes adstocked advertising values by applying a geometrically
    weighted sum of past advertising values, with the adstock_rate controlling the
    rate of decay.

    Args:
        x (np.ndarray): A 1D input array representing advertising values.
        adstock_rate (float): The adstock rate (decay rate) between 0 and 1.
        L (int): The length of the weights vector, determining the look-back period.

    Returns:
        np.ndarray: A 1D array of adstocked advertising values, with the same shape as the input array.
    """
    weights = jnp.array([adstock_rate**alpha for alpha in range(L)])
    shifted_x = jnp.stack([jnp.concatenate([jnp.zeros(l), x[:x.shape[0]-l]]) for l in range(L)])
    return jnp.dot(weights, shifted_x)

def media_transform(x, decay, beta, slope, L):
    """
    This function performs media transformation by applying adstock and response curve calculations on the input data.

    Parameters:
    x (numpy.ndarray): A 1-dimensional array representing the input data to be transformed.
    decay (float): A float value representing the adstock decay rate, which determines the rate at which the impact of the input data diminishes over time.
    beta (float): A float value representing the saturation parameter in the response curve, which controls the diminishing returns effect as input data increases.
    slope (float): A float value representing the slope of the response curve, which determines the overall scale of the transformed data.
    L (int): An integer representing the maximum lag in the adstock function, which determines the range of the adstock calculation.

    Returns:
    numpy.ndarray: A 1-dimensional array representing the transformed input data after applying adstock and response curve calculations.
    """
    adstock = adstocked_advertising_jnp(x, decay, L)
    saturation = response_curve_hyper(adstock, beta)
    return saturation*slope

def media_transform_vectorised(x, decay_vec, beta_vec, slope_vec, L):
    """
    This function provides a vectorized version of the media_transform function, allowing for efficient parallel computation across multiple sets of parameters.

    Parameters:
    x (numpy.ndarray): A 2-dimensional array representing the input data to be transformed.
    decay_vec (numpy.ndarray): A 1-dimensional array of float values representing the adstock decay rates for multiple transformations.
    beta_vec (numpy.ndarray): A 1-dimensional array of float values representing the saturation parameters for multiple response curves.
    slope_vec (numpy.ndarray): A 1-dimensional array of float values representing the slopes for multiple response curves.
    L (int): An integer representing the maximum lag in the adstock function, which determines the range of the adstock calculation.

    Returns:
    numpy.ndarray: A 2-dimensional array representing the transformed input data for each combination of decay, beta, and slope parameters.
    """
    return vmap(media_transform, (1, 0, 0, 0, None), 1)

def response_curve_hyper(x, beta, alpha=1.):
    """
    This function calculates the response curve using a hyperbolic function, which models the diminishing returns effect as input data increases.

    Parameters:
    x (numpy.ndarray or float): A 1-dimensional array or a float value representing the input data for the response curve calculation.
    beta (float): A float value representing the saturation parameter, which controls the diminishing returns effect as input data increases.
    alpha (float, optional): A float value representing the maximum response value when x approaches infinity, with a default value of 1.0.

    Returns:
    numpy.ndarray or float: A 1-dimensional array or a float value representing the response curve values for the given input data and parameters.
    """
    return alpha-beta/(x+(beta/alpha))
