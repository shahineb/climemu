import jax
import jax.numpy as jnp
import equinox as eqx
import healpy as hp
from functools import partial


@eqx.filter_jit
def haversine_distance_single(latlon1, latlon2, eps=1e-10):
    latlon1, latlon2 = jnp.deg2rad(latlon1), jnp.deg2rad(latlon2)
    dlat = latlon2[0] - latlon1[0] + eps
    dlon = latlon2[1] - latlon1[1] + eps
    coslat1 = jnp.cos(latlon1[0])
    coslat2 = jnp.cos(latlon2[0])
    coslat1_coslat2 = coslat1 * coslat2
    a = jnp.sin(dlat / 2)**2 + coslat1_coslat2 * jnp.sin(dlon / 2)**2
    c = 2 * jnp.asin(jnp.sqrt(a + eps).clip(max=1 - eps)) * 6371.0
    return c


@eqx.filter_jit
def haversine_distance_vmap(latlon, latlon_batch, eps=1e-10):
    haversine = partial(haversine_distance_single, latlon1=latlon, eps=eps)
    return jax.vmap(haversine)(latlon2=latlon_batch)


@eqx.filter_jit
def get_neighbourhood_mask(coords, latlon, threshold):
    dist_matrix = haversine_distance_vmap(coords, latlon)
    mask = dist_matrix < threshold
    return mask


@eqx.filter_jit
def get_neighbourhoods_masks(latlon, threshold):
    foo = partial(get_neighbourhood_mask, latlon=latlon, threshold=threshold)
    return jax.vmap(foo)(latlon)


@eqx.filter_jit
def get_k_nearest_neighbours_single(coords, latlon, k):
    dist_matrix = haversine_distance_vmap(coords, latlon)
    indices = jnp.argpartition(dist_matrix, k)[:k]
    return indices


@eqx.filter_jit
def get_k_nearest_neighbours(latlon1, latlon2, k):
    foo = partial(get_k_nearest_neighbours_single, latlon=latlon2, k=k)
    return jax.vmap(foo)(latlon1)


def compute_distance_threshold_edges(lat, lon, dist):
    latlon = jnp.stack(jnp.meshgrid(lat, lon)).reshape(2, -1).T
    A = get_neighbourhoods_masks(latlon, dist)
    nodes = jnp.arange(len(latlon))
    edges = jnp.stack(jnp.nonzero(A)).T
    return nodes, edges


def compute_latlon_to_healpix_edges(lat, lon, nside, k):
    # Make equiangualr coordinate grid
    latlon = jnp.array(jnp.meshgrid(lat, lon)).T.reshape(-1, 2)
    latlon_nodes = jnp.arange(len(latlon))

    # Make healpix coordinate grid
    npix = hp.nside2npix(nside)
    θ, φ = hp.pix2ang(nside, jnp.arange(npix), nest=True)
    lathp = 90 - jnp.rad2deg(θ)
    lonhp = jnp.rad2deg(φ)
    latlonhp = jnp.column_stack((lathp, lonhp))
    healpix_nodes = jnp.arange(npix)

    # Get k nearest neighbours both ways
    knn_indices_latlon2healpix = get_k_nearest_neighbours(latlonhp, latlon, k)  # (npix, k closest latlon)
    knn_indices_healpix2latlon = get_k_nearest_neighbours(latlon, latlonhp, k)  # (nlat*nlon, k closest healpix)

    # Create edges arrays - target node comes first
    edges_to_healpix = jnp.stack((jnp.repeat(healpix_nodes, k),
                                  knn_indices_latlon2healpix.flatten()), axis=-1)  # (npix*k, 2)
    edges_to_latlon = jnp.stack((jnp.repeat(latlon_nodes, k),
                                  knn_indices_healpix2latlon.flatten()), axis=-1)  # (nlat*nlon*k, 2)
    return edges_to_healpix, edges_to_latlon
    
