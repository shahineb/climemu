"""
Code taken from https://github.com/NVlabs/earth2grid/tree/main adapted to work with JAX
"""
from dataclasses import dataclass
from enum import Enum
from math import sqrt
import jax
import jax.numpy as jnp
import equinox as eqx


class Compass(Enum):
    """Cardinal directions in counter clockwise order"""
    S = 0
    E = 1
    N = 2
    W = 3


@dataclass(frozen=True)
class XY:
    """
    Assumes
        - i = n * n * f + n * y + x
        - the origin (x,y)=(0,0) is South
        - if clockwise=False follows the hand rule::

            Space
            |
            |
            |  / y
            | /
            |/______ x

        (Thumb points towards Space, index finger towards x, middle finger towards y)
    """
    origin: Compass = Compass.S
    clockwise: bool = False


HEALPIX_PAD_XY = XY(origin=Compass.N, clockwise=True)


@eqx.filter_jit
def local2xy(nside: int, x: jax.Array, y: jax.Array, face: jax.Array) -> jax.Array:
    """Convert a local x, y coordinate in a given face (S origin) to a global pixel index

    The local coordinates can be < 0 or > nside

    This can be used to implement padding

    Args:
        nside: int
        x: local coordinate [-nside, 2 * nside), origin=S
        y: local coordinate [-nside, 2 * nside), origin=S
        face: index of the face [0, 12)
        right_first: if True then traverse to the face in the x-direction first

    Returns:
        x, y, f:  0 <= x,y< nside. f>12 are the missing faces.
            See ``_xy_with_filled_tile`` and ``pad`` for the original hpxpad
            methods for filling them in.
    """
    # adjacency graph (8 neighbors, counter-clockwise from S)
    # Any faces > 11 are missing
    neighbors = jnp.array(
        [
            # pole
            [8, 5, 1, 1, 2, 3, 3, 4],
            [9, 6, 2, 2, 3, 0, 0, 5],
            [10, 7, 3, 3, 0, 1, 1, 6],
            [11, 4, 0, 0, 1, 2, 2, 7],
            # equator
            [16, 8, 5, 0, 12, 3, 7, 11],
            [17, 9, 6, 1, 13, 0, 4, 8],
            [18, 10, 7, 2, 14, 1, 5, 9],
            [19, 11, 4, 3, 15, 2, 6, 10],
            # south pole
            [10, 9, 9, 5, 0, 4, 11, 11],
            [11, 10, 10, 6, 1, 5, 8, 8],
            [8, 11, 11, 7, 2, 6, 9, 9],
            [9, 8, 8, 4, 3, 7, 10, 10],
        ]
    )

    # number of left turns the path takes while traversing from face i to j
    turns = jnp.array(
        [
            # pole
            [0, 0, 0, 3, 2, 1, 0, 0],
            # equator
            [0, 0, 0, 0, 0, 0, 0, 0],
            # south pole
            [2, 1, 0, 0, 0, 0, 0, 3],
        ]
    )
    # x direction
    face_shift_x = x // nside
    face_shift_y = y // nside

    # TODO what if more face_shift_x, face_shift_y = 2, 1 or similar?
    # which direction should we traverse faces in?
    direction_lookup = jnp.array([[0, 7, 6], [1, -1, 5], [2, 3, 4]])

    direction = direction_lookup[face_shift_x + 1, face_shift_y + 1]
    new_face = jnp.where(direction != -1, neighbors[face, direction], face)
    origin = jnp.where(direction != -1, turns[face // 4, direction], 0)

    # rotate back to origin = S convection
    for i in range(1, 4):
        nx, ny = _rotate(nside, i, x, y)
        x = jnp.where(origin == i, nx, x)
        y = jnp.where(origin == i, ny, y)

    face = new_face
    return x % nside, y % nside, face


@eqx.filter_jit
def local2local(nside: int, src: XY, dest: XY, x: jax.Array, y: jax.Array):
    """Convert a local index (x, y) between different XY conventions"""
    if src == dest:
        return x, y
    rotations = src.origin.value - dest.origin.value
    x, y = _rotate(nside=nside, rotations=-rotations if dest.clockwise else rotations, x=x, y=y)

    if src.clockwise != dest.clockwise:
        x, y = y, x

    return x, y


@eqx.filter_jit
def _rotate(nside: int, rotations: int, x, y):
    """rotate (x,y) counter clockwise"""
    k = rotations % 4
    # Apply the rotation based on k
    if k == 1:  # 90 degrees counterclockwise
        return nside - y - 1, x
    elif k == 2:  # 180 degrees
        return nside - x - 1, nside - y - 1
    elif k == 3:  # 270 degrees counterclockwise
        return y, nside - x - 1
    else:  # k == 0, no change
        return x, y
    

PIXEL_ORDER = XY()


@eqx.filter_jit
def _xy_with_filled_tile(nside, x1, y1, f1):
    """Handles an points with missing tile information following the HPXPAD strategy

    Missing tiles are defined for face >= 12. 12-16 are the N missing tiles, and
    16-20 the south missing tiles (from W to east).

    Since there is an ambiguity return both x and y.
    """

    # handle missing tiles
    # for N tiles
    # f(x, y) is filled by shuffling from the left
    # case x > y: (x, y) -> (y, )
    # examples  (for nside = 4)
    #   (3, 1)-> (0, 1)
    #   (3, 2) -> (1, 2)
    #   (3, 3) -> (2, 3)
    # generalize
    #   (i, j)-> (i + j, j)  in the missing face
    #   (i' - j, j) -> (i', j)

    is_missing_n_pole_tile = (f1 >= 12) & (f1 < 16)
    west_face = jnp.where(is_missing_n_pole_tile, f1 - 13, 0) % 4
    east_face = (west_face + 1) % 4

    # two sets of indices
    def _pad_from_west(x1, y1, west_face):
        f_west = jnp.where(is_missing_n_pole_tile & (x1 <= y1), west_face, f1)
        x_west = jnp.where(is_missing_n_pole_tile & (x1 < y1), (x1 - y1) % nside, x1)
        x_west = jnp.where(is_missing_n_pole_tile & (x1 == y1), nside - 1, x_west)
        y_west = y1
        return x_west, y_west, f_west

    x_west, y_west, f_west = _pad_from_west(x1, y1, west_face)
    y_east, x_east, f_east = _pad_from_west(y1, x1, east_face)

    # S pole
    is_missing_s_pole_tile = (f1 >= 16) & (f1 < 20)
    east_face = (f1 - 16) % 4 + 8
    west_face = (east_face - 9) % 4 + 8

    # two sets of indices
    def _pad_from_east(x1, y1, east_face, f1):
        """Test cases

        (1, 0) -> (0, 0)
        (3, 2) -> (0, 2)
        """
        f_west = jnp.where(is_missing_s_pole_tile & (x1 >= y1), east_face, f1)
        # x_west = torch.where(is_missing_s_pole_tile & (x1 > y1), 1(x1-y1) %nside, x1)
        x_west = jnp.where(is_missing_s_pole_tile & (x1 > y1), (x1 - y1 - 1) % nside, x1)
        x_west = jnp.where(is_missing_s_pole_tile & (x1 == y1), 0, x_west)
        y_west = y1
        return x_west, y_west, f_west

    x_west, y_west, f_west = _pad_from_east(x_west, y_west, east_face, f_west)
    y_east, x_east, f_east = _pad_from_east(y_east, x_east, west_face, f_east)
    return (x_west, y_west, f_west), (x_east, y_east, f_east)


@eqx.filter_jit
def pad_with_dim(x, padding, dim=1, pixel_order=XY()):
    """
    x[dim] is the spatial dim
    """
    dim = dim % x.ndim
    pad = padding
    npix = x.shape[dim]
    nside = int(sqrt(npix / 12))

    # setup padded grid
    i = jnp.arange(-pad, nside + pad)
    j = jnp.arange(-pad, nside + pad)
    f = jnp.arange(12)
    f, j, i = jnp.meshgrid(f, j, i, indexing="ij")

    # convert these ponints to origin=S, clockwise=False order
    # (this is the order expected by local2xy and _xy_with_filled_tile)
    i, j = local2local(nside, pixel_order, PIXEL_ORDER, i, j)

    # get indices in source data for target points
    i1, j1, f1 = local2xy(nside, i, j, f)

    (i1, j1, f1), (i2, j2, f2) = _xy_with_filled_tile(nside, i1, j1, f1)
    # convert these back to ``pixel_order`` since we will be grabbing
    # data from ``x`` in this order
    i1, j1 = local2local(nside, PIXEL_ORDER, pixel_order, i1, j1)
    i2, j2 = local2local(nside, PIXEL_ORDER, pixel_order, i2, j2)

    # prepare final flat indexes
    f1 = jnp.where(f1 < 12, f1, -1)
    f2 = jnp.where(f2 < 12, f2, -1)
    xy_west = jnp.ravel(f1 * (nside * nside) + j1 * nside + i1)
    xy_east = jnp.ravel(f2 * (nside * nside) + j2 * nside + i2)

    shape = [1] * x.ndim
    shape[dim] = xy_west.size

    # average the potential ambiguous regions
    padded_from_west = jnp.where(xy_west.reshape(shape) >= 0, _take(x, xy_west, dim), 0)
    padded_from_east = jnp.where(xy_east.reshape(shape) >= 0, _take(x, xy_east, dim), 0)
    denom = jnp.where(xy_west >= 0, 1, 0) + jnp.where(xy_east >= 0, 1, 0)

    return (padded_from_east + padded_from_west) / denom.reshape(shape)


def _take(x, index, dim):
    return jnp.take(x, index, axis=dim)


@eqx.filter_jit
def pad(x, padding):
    """A padding function compatible with healpixpad inputs

    Args:
        x: (c, f, h, w)
        padding: int
    """
    c, f, nside, _ = x.shape
    x = x.reshape(c, f * nside**2)
    x = pad_with_dim(x, padding, dim=-1, pixel_order=HEALPIX_PAD_XY)
    x = x.reshape(c, f, nside + 2 * padding, nside + 2 * padding)
    return x