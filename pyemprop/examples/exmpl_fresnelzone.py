
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from ..propagation import mcaprop
from ..utils.geometry import circle
from ..utils.scalarwave import getAmp
from ..utils.optel import fresnZoneDist
from ..utils.plotting import plotsingle

def run():
    # Settings  ________
    shape = (2048, 2048)
    samp = 10.0 # um/px
    wln = 632.8 * (10**(-3)) # nm -> um; He-Ne laser

    # Circle  ________
    R = 1.0 * (10**(3)) # mm -> um
    circ = circle(shape, samp, R)
    pha = np.zeros(circ.shape, circ.dtype)
    u1 = circ*np.exp(1j*pha)

    # Propagation  ________
    # Fresnel zone distance:
    propdist1 = fresnZoneDist(R, wln, 1)
    propdist2 = fresnZoneDist(R, wln, 2)

    u2 = mcaprop(
        u1, propdist1, samp, wln
        , onaxis=True
        # ,padding=(int(u1.shape[0]/2),int(u1.shape[1]/2))
    )
    u3 = mcaprop(
        u1, propdist2, samp, wln
        , onaxis=True
        # ,padding=(int(u1.shape[0]/2),int(u1.shape[1]/2))
    )

    # Display  ________
    fig, axs = plt.subplots(1, 3, layout="constrained")
    fig.suptitle("Fresnel zone example")
    # pos0 = axs[0].imshow(circ, cmap='hot', interpolation='nearest')
    plotsingle(circ,samp, ax=axs[0], marg=-0.4, labls=("x [px]", "y [px]"))
    axs[0].set_title("Input shape")

    # axs[1].imshow((getAmp((u2))), cmap='hot', interpolation='nearest')
    plotsingle(getAmp(u2),samp, ax=axs[1], marg=-0.4, labls=("x [px]", "y [px]"))
    axs[1].set_title(f"1 Fresnel zone @ {propdist1/1000:.2f} mm")

    # axs[2].imshow((getAmp((u3))), cmap='hot', interpolation='nearest')
    plotsingle(getAmp(u3),samp, ax=axs[2], marg=-0.4, labls=("x [px]", "y [px]"))
    axs[2].set_title(f"2 Fresnel zones @ {propdist2/1000:.2f} mm")

    plt.show()