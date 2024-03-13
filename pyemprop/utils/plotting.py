# plotting utilities
from matplotlib import pyplot as plt
import numpy as np


def _setup_tics(subax, samp, xz=False, zsamp=None, r0=None, tickpos=None):
    xlims = subax.get_xlim()
    ylims = subax.get_ylim()

    if r0 is not None:
        (x0,y0) = r0
    else:
        x0 = (xlims[1]+xlims[0])/2
        y0 = (ylims[1]+ylims[0])/2
    
    if tickpos == None:
        pos = [0.1, 0.3, 0.5, 0.7, 0.9]
        xpos = [xp*(xlims[1]-xlims[0])+xlims[0] for xp in pos]
        ypos = [yp*(ylims[1]-ylims[0])+ylims[0] for yp in pos]
    else:
        (xpos,ypos) = tickpos

    if xz:
        xlab = [int((xp-x0)*samp) for xp in xpos]
        ylab = [int((yp)*zsamp) for yp in ypos]
    else:
        xlab = [int((xp-x0)*samp) for xp in xpos]
        ylab = [int((yp-y0)*samp) for yp in ypos]

    plt.setp(subax, xticks=xpos, xticklabels=xlab, yticks=ypos, yticklabels=ylab)

def crop(ar, samp, croprange):
    croprange = np.asarray(croprange)/samp
    return ar[
            np.floor(ar.shape[0]/2-croprange[0]/2).astype(np.int64):np.ceil(ar.shape[0]/2+croprange[0]/2).astype(np.int64)
            , np.floor(ar.shape[1]/2-croprange[1]/2).astype(np.int64):np.ceil(ar.shape[1]/2+croprange[1]/2).astype(np.int64)
        ]

def plotsingle(img, samp, ax=None, marg=0.0, tickpos=None, save=None, logger=None, cbar=True, labls=None, extent=None, croprange=None, aspect=None):
    if ax is None:
        fig, ax = plt.subplots(1,1
                            ,layout="constrained"
                            )
    if croprange is not None:
        imgplt = ax.imshow(crop(img.astype(np.float64), samp, croprange=croprange), cmap='hot', interpolation='nearest', extent=extent, aspect=aspect)
    else:
        imgplt = ax.imshow(img, cmap='hot', interpolation='nearest', extent=extent, aspect=aspect)
    # ax[0].set_title("Amplitude of the object")
    if labls is not None:
        ax.set_xlabel(labls[0])
        ax.set_ylabel(labls[1])
    if cbar:
        plt.colorbar(imgplt, shrink=0.5)
    ax.margins(marg)

    _setup_tics(
        ax, samp, tickpos=tickpos
    )
    if save is not None:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        if logger is not None:
            logger.info(f"Plot saved to: {save}.")