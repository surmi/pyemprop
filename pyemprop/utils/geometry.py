import numpy as np
import numpy.typing as npt


def coord(shp,samp):
    # coordinate system with 0 at the center (for even shape 0 is in n/2+1).
    xlin = np.linspace(-shp[0]/2, shp[0]/2 - 1, num=shp[0])*samp
    ylin = np.linspace(-shp[1]/2, shp[1]/2 - 1, num=shp[1])*samp
    x, y = np.meshgrid(xlin, ylin)
    return (x,y)


def distcent(x,y):
    r = np.sqrt(x**2+y**2)
    return r


def circle(shp, samp, R, x0=0, y0=0, fill=1.0):
    res = np.zeros(shp) 
    
    x = np.linspace(
        -int(res.shape[0]/2), int(res.shape[0]/2 -1), res.shape[0]
    ) * samp
    y = np.linspace(
        -int(res.shape[1]/2), int(res.shape[1]/2 -1), res.shape[1]
    ) * samp

    xx, yy = np.meshgrid(x, y)

    circmask = np.sqrt((xx-x0)**2 + (yy-y0)**2) <= R
    res[circmask] = 1.0

    return res


def repeatShape(
    tilemat: npt.ArrayLike, 
    matsize: tuple[(int, int)], 
    mode="fill2matching",
    **kwargs
):
    """_summary_

    Args:
        shape (npt.ArrayLike): array with shape to be repeated
        period (int): _description_
        matsize (tuple[): _description_
    """
    # tilemat is square
    # matsize is multiple of tilemat size - otherwise different mode than default
    # period is in points (indices)
    # period is int
    # TODO different period in different directions
    # padding for tiles is added at the end of rows and columns

    
    period = kwargs.get('period', None)
    # make a matrix of specific size - zeros
    res = np.zeros(matsize, dtype=tilemat.dtype)

    # if provided, set period of repetition (pad shape that would be repeated)
    if period is not None:
        # pad shape before repeating the shape
        padsize = period - tilemat.shape[0]
        tilemat = np.pad(tilemat, ((0,padsize),(0,padsize)), "constant", constant_values=(0,))

    # fill depending on mode
    if mode == "fill2matching":
        # fill resulting matrix with tiles up to exact multiple of tile size
        rowrep = int(res.shape[0]/tilemat.shape[0])
        colrep = int(res.shape[1]/tilemat.shape[1])
        tiles = np.tile(tilemat, (rowrep, colrep))
        shapetiled = tiles.shape
        slicex = slice(
            round((res.shape[0]-shapetiled[0])/2),
            round((res.shape[0]-shapetiled[0])/2) + shapetiled[0],
            1
        )
        slicey = slice(
            round((res.shape[1]-shapetiled[1])/2),
            round((res.shape[1]-shapetiled[1])/2) + shapetiled[1],
            1
        )
        res[slicex,slicey] = tiles
    elif mode == "fillandcrop":
        # if there is no exact multiple of tile size in required matrix size
        # make a temporary matrix with size allowing to fit celling of
        # (matrix size)/(tile size) and then crop matrix to the size
        # TODO finish
        pass
    else:
        raise ValueError("Variable 'mode' does not accept such value.")
    
    return res


def sqrtilepx(
    tileshape:tuple[(int, int)],
    squaresize:int,
    fill:float=1.0
):
    tile = np.zeros(tileshape, dtype=np.float64)
    begsqrow = int((tileshape[0] - squaresize)/2)
    endsqrow = int((tileshape[0] - squaresize)/2 + squaresize)
    begsqcol = int((tileshape[0] - squaresize)/2)
    endsqcol = int((tileshape[0] - squaresize)/2 + squaresize)
    tile[begsqrow:endsqrow, begsqcol:endsqcol] = fill

    return tile