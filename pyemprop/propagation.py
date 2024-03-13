# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from tqdm import tqdm

from .utils.scalarwave import kk


def subprodist(msize, samp, wln):
    return msize*samp**2/wln # 2*msize if passed without additional padding size


def fstf(vx, vy, wln, dist, n0=1):
    # Free space transfer function
    # according to S. Bahaa and T. Malvin, Fundamentals of Photonics, 
    # 3rd ed., 2019, page 119 but without negative sign and with n0.
    # return np.exp(-1j*2*np.pi*dist*
    #     np.emath.sqrt(np.square(n0/wln)-np.square(vx)-np.square(vy))
    #     # np.sqrt(np.square(n0/wln+0j)-np.square(vx)-np.square(vy))
    #     )
    return np.power(
        np.exp(-1j*2*np.pi*dist)
        , np.emath.sqrt(np.square(n0/wln)-np.square(vx)-np.square(vy))
    )


def fstfonaxis(vx, vy, wln, dist):
    # Free space transfer function with Fresnel (paraxial) approx.
    # according to S. Bahaa and T. Malvin, Fundamentals of Photonics, 
    # 3rd ed., 2019, page 121 but without negative sign and with n0.
    k = kk(wln)
    H0 = np.exp(-1j*k*dist)
    return H0*np.exp(1j*np.pi*wln*dist*(np.square(vx)+np.square(vy)))


def fsir(x, y, wln, dist):
    # free space impuls response
    # according to Sypek 1995
    k = 2*np.pi/wln
    h0 = np.exp(1j*k*dist)/(1j*wln*dist)
    return h0*np.exp(1j*k/(2*dist)*(np.square(x)+np.square(y)))


def propagation(u1, wln, dist, samp, onaxis=False):    
    
    U1 = fftshift(fft2(ifftshift(u1)))
    # generate x and y grid
    dfx = 1/u1.shape[0]/samp
    dfy = 1/u1.shape[1]/samp
    xlin = np.linspace(-U1.shape[0]/2, U1.shape[0]/2 - 1, num=U1.shape[0])*dfx
    ylin = np.linspace(-U1.shape[1]/2, U1.shape[1]/2 - 1, num=U1.shape[1])*dfy
    vx, vy = np.meshgrid(xlin, ylin)
    if onaxis:
        H = fstfonaxis(vx, vy, wln, dist)
    else:
        H = fstf(vx, vy, wln, dist)
    U2 = np.multiply(U1, H)
    
    # xlin = np.linspace(-u1.shape[0]/2, u1.shape[0]/2 - 1, num=u1.shape[0])*samp
    # ylin = np.linspace(-u1.shape[1]/2, u1.shape[1]/2 - 1, num=u1.shape[1])*samp
    # x, y = np.meshgrid(xlin, ylin)

    # h = fsir(x, y, wln, dist)
    # U2 = np.multiply(U1, fftshift(fft2(ifftshift(h))))

    return fftshift(ifft2(ifftshift(U2)))


def propnoshift(u1, wln, dist, samp, onaxis=False, padding=None):    
    
    U1 = fftshift(fft2((u1)))
    # generate x and y grid
    dfx = 1/u1.shape[0]/samp
    dfy = 1/u1.shape[1]/samp
    xlin = np.linspace(-U1.shape[0]/2, U1.shape[0]/2 - 1, num=U1.shape[0])*dfx
    ylin = np.linspace(-U1.shape[1]/2, U1.shape[1]/2 - 1, num=U1.shape[1])*dfy
    vx, vy = np.meshgrid(xlin, ylin)
    if onaxis:
        H = fstfonaxis(vx, vy, wln, dist)
    else:
        H = fstf(vx, vy, wln, dist)
    U2 = np.multiply(U1, H)
    
    # xlin = np.linspace(-u1.shape[0]/2, u1.shape[0]/2 - 1, num=u1.shape[0])*samp
    # ylin = np.linspace(-u1.shape[1]/2, u1.shape[1]/2 - 1, num=u1.shape[1])*samp
    # x, y = np.meshgrid(xlin, ylin)

    # h = fsir(x, y, wln, dist)
    # U2 = np.multiply(U1, fftshift(fft2(ifftshift(h))))

    return (ifft2(ifftshift(U2)))


def mcaprop(inmat, dist, samp, wln, onaxis=False, padding=None, logger=None):
    # assumptions: 2d matrix on the input, matrix is square

    if logger is not None:
        logger.info(f"Starting propagation on distance {dist} mm")
    # padding
    if padding is None:
        u1 = inmat.copy()
    else:
        u1 = np.pad(inmat, padding)
    
    # TODO: memory restrictions due to the size of padded matrix?

    # calculate subpropagation distances
    spd = subprodist(float(min(u1.shape)), samp, wln)
    # logic for subpropagion/propagation
    propsteps = []
    if spd < dist:
        # prepare subpropagations
        spn = math.floor(dist/spd)# number of full propagations

        propsteps.extend([spd]*(spn))
        propsteps.extend([dist-(spd*spn)]) # "lefover" propagation
    else:
        propsteps.append(dist)
    if logger is not None:
        logger.info(f"Propagation in {len(propsteps)} step(s).")

    #propagate
    u2 = np.zeros(u1.shape, u1.dtype)
    u1 = ifftshift(u1)
    for d in tqdm(propsteps):
        u2 = propnoshift(u1,wln,d,samp,onaxis=onaxis)
        u1 = u2.copy()
    u2 = fftshift(u2)
    slicex = slice(
            round((u2.shape[0]-inmat.shape[0])/2),
            round((u2.shape[0]-inmat.shape[0])/2) + inmat.shape[0],
            1
        )
    slicey = slice(
            round((u2.shape[1]-inmat.shape[1])/2),
            round((u2.shape[1]-inmat.shape[1])/2) + inmat.shape[1],
            1
        )

    if logger is not None:
        logger.info(f"Propagation finished.")
    
    return u2[slicex, slicey]
    

def testmeth():
    print("test")