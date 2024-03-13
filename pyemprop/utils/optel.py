import numpy as np
import numpy.typing as npt
from typing import Tuple

def fresnZoneDist(R:float, wln:float, m:int) -> float:
    """Distance at which contribution of `m` Fresnel
    zones can be observed after circular aperture.
    
    Care to keep the same units for `R` and `wln`.

    Args:
        R (float): radius of tested circular aperture
        wln (float): wavelength
        m (int): number of required Fresnel zones

    Returns:
        float: distance
    """    
    return R**2/(m*wln) - wln/4*m


def talb_dist(wln: float, d: float, C:int=1) -> float:
    """Returns full Talbot distance. Units of `wln` and
    `d` have to be the same.

    Args:
        wln (float): wavelength of radiation
        d (float): period (distance)
        C (int, optional): constant. Defaults to 1.

    Returns:
        float: Talbot distance
    """    
    return 2*d*d/wln*C


def lensParax(shp: Tuple[int, int], samp: float, f: Tuple[float, float], wln: float, x0:float=0, y0:float=0) -> npt.NDArray[np.float64]:
    """Phase modulation of lens with paraxial approximation

    Args:
        shp (Tuple[int, int]): shape of the calculation matrix
        samp (float): sampling distance
        f (Tuple[float, float]): focal length (along both axis)
        wln (float): wavelength of the radiation
        x0 (float, optional): x position of the center of the lens. Defaults to 0.
        y0 (float, optional): y position of the center of the lens. Defaults to 0.

    Returns:
        npt.NDArray[np.float64]: phase modulation of a lens
    """
    res = np.zeros(shp)

    (f1, f2) = f

    x = np.linspace(-int(res.shape[0]/2), int(res.shape[0]/2 -1), res.shape[0]) * samp
    y = np.linspace(-int(res.shape[1]/2), int(res.shape[1]/2 -1), res.shape[1]) * samp

    xx, yy = np.meshgrid(x, y)

    return np.exp(-1j*np.pi/wln* ((xx-x0)**2/f1 + (yy-y0)**2/f2))


def lensNonParax(shp: Tuple[int, int], samp: float, f: Tuple[float, float], wln: float, x0:float=0, y0:float=0) -> npt.NDArray[np.float64]:
    """Phase modulation of lens without paraxial approximation

    Args:
        shp (Tuple[int, int]): shape of the calculation matrix
        samp (float): sampling distance
        f (Tuple[float, float]): focal length (along both axis)
        wln (float): wavelength of the radiation
        x0 (float, optional): x position of the center of the lens. Defaults to 0.
        y0 (float, optional): y position of the center of the lens. Defaults to 0.

    Returns:
        npt.NDArray[np.float64]:  phase modulationof a lens
    """
    res = np.zeros(shp)

    x = np.linspace(-int(res.shape[0]/2), int(res.shape[0]/2 -1), res.shape[0]) * samp
    y = np.linspace(-int(res.shape[1]/2), int(res.shape[1]/2 -1), res.shape[1]) * samp

    xx, yy = np.meshgrid(x, y)
    r = np.sqrt((x0-xx)**2+(y0-yy)**2)

    if f < 0:
        return np.exp(1j*2*np.pi/wln* np.sqrt(f**2+np.power(r, 2)))
    else:
        return np.exp(-1j*2*np.pi/wln* np.sqrt(f**2+np.power(r, 2)))