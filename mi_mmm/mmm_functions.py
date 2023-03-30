import jax.numpy as jnp

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
