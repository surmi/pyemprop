import numpy as np
import numpy.typing as npt

def getAmp(u:npt.NDArray):
    return np.abs(u)


def getInt(u:npt.NDArray):
    return getAmp(u)**2


def getPha(u:npt.NDArray):
    return np.angle(u)


def pw2sphcor(p: float, z: float):
    """correction of image distance due to change of wavefront
    from pw to spherical (diffractive scale theorem for
    Fresnel diffraction)

    Args:
        p (float): spherical wavefront radius of curvature
        z (float): image distance for pw illuminaiton

    Returns:
        float: corrected image distance
    """
    return p*z/(p-z)


def sph2pwcor(p: float,q: float):
    """correction of image distance due to change of wavefront
    from spherical to pw (diffractive scale theorem for
    Fresnel diffraction)

    Args:
        p (float): spherical wavefront radius of curvature
        q (float): image distance for spherical illuminaiton

    Returns:
        float: corrected image distance
    """
    return p*q/(p+q)


# Gaussian beam-----------------------------------
def zr2w0(zr: float, wln: float, n:float=1) -> float:
    """Rayleigh range to waist radius.

    Args:
        zr (float): Rayleigh range
        wln (float): radiation wavelength
        n (float, optional): Refractive index of medium. Defaults to 1.

    Returns:
        float: waist radius
    """
    return np.sqrt(zr*wln/(np.pi*n))


def zr2w0k(zr:float, k:float, n:float=1) -> float:
    """Rayleigh range to waist radius. Based on wavenumber of the radiation.

    Args:
        zr (float): Rayleigh range
        k (float): radiation wavenumber
        n (float, optional): Refractive index of medium. Defaults to 1.

    Returns:
        float: waist radius
    """
    return np.sqrt(2*zr/(k*n))


def w02zr(w0:float, wln:float, n:float=1) -> float:
    """Waist radius to Rayleigh range.

    Args:
        w0 (float): waist radius
        wln float): radiation wavelength
        n (float, optional): Refractive index of medium. Defaults to 1.

    Returns:
        float: Rayleigh range
    """
    return np.pi*w0*w0*n/wln


def w02zrk(w0:float, k:float, n:float=1) -> float:
    """Waist radius to Rayleigh range. Based on wavenumber of the radiation.

    Args:
        w0 (float): waist radius
        k float): radiation wavenumber
        n (float, optional): Refractive index of medium. Defaults to 1.

    Returns:
        float: Rayleigh range
    """
    return k*w0*w0*n/2


def gouy(z:float, zr:float, p0:float=0) -> float:
    """Gouy phase shift.

    Args:
        z (float): position along the propagation axis
        zr (float): Rayleigh range
        p0 (float, optional): position of the waist. Defaults to 0.

    Returns:
        float: Gouy phase shift
    """
    return np.arctan2((z-p0), zr)


def w(w0:float, z:float, zr:float, p0:float=0) -> float:
    """Beam radius evolution due to diffraction.

    Args:
        w0 (float): waist radius
        z (float): position along the propagation axis
        zr (float): Rayleigh range
        p0 (float, optional): position of the waist. Defaults to 0.

    Returns:
        float: beam radius
    """
    return w0*np.sqrt(1+((z-p0)/zr)**2)


def R(w0: float, z:float, zr:float, p0:float=0) -> float:
    """Wavefront's radius of curvature.

    Args:
        w0 (float): waist radius
        z (float): position along the propagation axis
        zr (float): Rayleigh range
        p0 (float, optional): position of the waist. Defaults to 0.

    Returns:
        float: radius of curvature
    """
    return(z-p0)*(1+ (zr/(z-p0))**2)


def kk(wln:float) -> float:
    """Wavelength to wavenumber

    Args:
        wln (float): wavelength

    Returns:
        float: wavenumber
    """
    return 2*np.pi/wln


def gaussianbeamk(w0:float, z:float, zr:float, r:npt.NDArray[np.float64], k:float, p0:float=0, E0:float=1) -> npt.NDArray[np.complex_]:
    """Gaussian beam amplitude distribution (electric field) with paraxial approximation.

    Args:
        w0 (float): waist radius
        z (float): position along the axis of propagation
        zr (float): Rayleigh range
        r (npt.NDArray[np.float64]): radial position in a plane perpendicular to the propagation axis
        k (float): wavenumber
        p0 (float, optional): position of the waist. Defaults to 0.
        E0 (float, optional): electric field amplitude at the origin. Defaults to 1.

    Returns:
        npt.NDArray[np.complex_]: amplitude at given position
    """
    #TODO there are some overflow errors (too large argument for exp)
    wz = w(w0,z,zr,p0)
    if z == 0:
        return E0*w0/wz* np.exp(-np.square(r)/wz**2)*1
    else:
        Rz = R(w0,z,zr,p0)
        Gz = gouy(z,zr,p0)
        return E0*w0/wz* np.exp(-np.square(r)/wz**2)*np.exp(-1j*(k*z+k*(r**2/(2*Rz))-Gz))


def qpar(z:float, zr:float) -> complex:
    """Complex beam parameter (also called "q-parameter")

    Args:
        z (float): position along the axis of propagation
        zr (float): Rayleigh range

    Returns:
        complex: q parameter
    """
    return z+1j*zr


def gaussianbeamq(z:float, zr:float, r:npt.NDArray[np.float64], wln:float) -> npt.NDArray[np.complex_]:
    """Gaussian beam calculated with q-parameter. Assumes circular beam.

    Args:
        z (float): position along the axis of propagation
        zr (float): Rayleigh range
        r (npt.NDArray[np.float64]): radial position in a plane perpendicular to the propagation axis
        wln (float): wavelength

    Returns:
        npt.NDArray[np.complex_]: Gaussian beam amplitude
    """
    qq = qpar(z,zr)
    k = kk(wln)
    return 1/qq*np.exp(-1j*k*np.square(r) /(2*qq))


def gaussianbeam(w0:float, z:float, zr:float, r:npt.NDArray[np.float64], wln:float, p0:float=0, E0:float=1) -> npt.NDArray[np.complex_]:
    """Wrapper for `gaussianbeamk()` that takes wavelength instead of wavenumber.
    
    Gaussian beam amplitude distribution (electric field) with paraxial approximation.

    Args:
        w0 (float): waist radius
        z (float): position along the axis of propagation
        zr (float): Rayleigh range
        r (npt.NDArray[np.float64]): radial position in a plane perpendicular to the propagation axis
        wln float): wavelength
        p0 (float, optional): position of the waist. Defaults to 0.
        E0 (float, optional): electric field amplitude at the origin. Defaults to 1.

    Returns:
        npt.NDArray[np.complex_]: amplitude at given position
    """
    kk = kk(wln)
    return gaussianbeamk(w0,z,zr,p0,r,kk,E0)


def gaussianint(w0:float, z:float, r:npt.NDArray[np.float64], wln:float, p0:float=0, I0:float=1) -> npt.NDArray[np.complex_]:
    """Intensity disribution of Gaussian beam at given positon.

    Args:
        w0 (float): waist radius
        z (float): position along the axis of propagation
        r (npt.NDArray[np.float64]): radial position in the plane perpendicular to the propagation axis
        wln (float): wavelength
        p0 (float, optional): position of the waist. Defaults to 0.
        I0 (float, optional): intensity at the origin. Defaults to 1.

    Returns:
        npt.NDArray[np.complex_]: intensity at given position
    """
    zr = w02zr(w0,wln)
    wz = w(w0,z,zr,p0)
    return I0*(w0/wz)**2*np.exp(-2*r**2/(wz*2))


def fwhm2w(fwhm:float) -> float:
    """Convert FWHM to beam radius

    Args:
        fwhm (float): FWHM of the beam

    Returns:
        float: beam radius of the beam
    """
    return fwhm/(np.sqrt(2*np.log(2)))


def w2z(w0:float, wz:float, zr:float) -> float:
    """Calculate postion from the waist along the axis of propagation

    Args:
        w0 (float): waist radius
        wz (float): radius at the z position
        zr (float): Rayleigh range

    Returns:
        float: position along the axis of propagation
    """
    return zr*np.sqrt((wz/w0)**2-1)


def srcgaus():
    # TODO: generation of Gaussian beam source (similar for others)
    
    pass