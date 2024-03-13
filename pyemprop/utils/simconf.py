"""Optional module providing utility of simulation configuration
as an object.
"""
import numpy as np
from numpy import typing
import pandas as pd
from typing import Iterable
from itertools import combinations,chain,filterfalse
from collections import abc

from .scalarwave import kk,gaussianbeamk,w02zr,getAmp,getPha
from .geometry import coord,distcent


class Simconf:
    """General class for definition of simulation parameters
    """    
    def __init__(self, title) -> None:
        # parmeters common for all simulations
        self._compars = {
            "title": title,
            "startdt": None,
            "stopdt": None
        }
        # units of the parameters
        self._units = {
            "title": "text",
            "startdt": "y-m-d h-m",
            "stopdt": "y-m-d h-m"
        }
        # descriptions of parameters (for reporting)
        self._desc = {
            "title": "Title",
            "startdt": "Date and time of simulation start",
            "stopdt": "Date and time of simulation end"
        }
    

    def __getattribute__(self, parnm):
        try:
            return object.__getattribute__(self, parnm)
        except AttributeError:
            try:
                return self._compars[parnm]
            except KeyError:
                print(f"There is no atribute named {parnm}")

    @property
    def units(self):
        """Units dictionary for defined properties. Keys are
        names of the parameters."""
        return self._units
    
    @units.setter
    def units(self, prop, unit):
        if prop in self._units:
            self._units.update({prop: unit})
        else: raise KeyError(f"{prop} is not one of the keys.")
    
    @units.deleter
    def units(self):
        del self._units

    @property
    def desc(self):
        """Dictionary of descriptions for parameters (used
        for documentation of simulaiton). Keys are names
        of the parameters."""
        return self._desc
    
    @desc.setter
    def desc(self, prop, strdesc):
        if prop in self._units:
            self._units.update({prop: strdesc})
        else: raise KeyError(f"{prop} is not one of the keys.")

    @desc.deleter
    def desc(self):
        del self._desc


class Scalarprop(Simconf):
    """Class that adds common parameters for simulation of
    scalar propagation.
    """    
    def __init__(self, title, mshape, sampd, wln, padd) -> None:
        super().__init__(title)
        self._compars.update({
            "mshape":mshape,
            "sampd":sampd,
            "wln":wln,
            "padd":padd
        })

        self._units.update({
            "mshape":"(px,px)",
            "sampd":"mm",
            "wln":"mm",
            "padd":"(px,px)"
        })

        self._desc.update({
            "mshape":"Calc. mat. shape",
            "sampd":"Sampling distance",
            "wln":"Radiation wavelength",
            "padd":"Calc. mat. padding for propagation"
        })


class Batchsim:
    """Base class for batch simulations.
    """
    def __init__(
            self,
            title:str,
            parnames:Iterable[str] | None,
            parlist:Iterable[list] | None,
            **kwargs
    ) -> None:
        """Create Batchsim object.

        Args:
            title (str): title of a simulation.
            parnames (Iterable[str]): list of parameter names.
            parlist (Iterable[list]): list of parameter values.
        """        
        self._batches = None
        units = {}; descs = {}
        self._pardict = {}

        if "units" in kwargs:
            units = kwargs["units"]
        if "desc" in kwargs:
            descs = kwargs["desc"]

        if parnames is not None and parlist is not None:
            self.addbatchpars(
                parvals=parlist,
                parnames=parnames,
                units=units,
                descs=descs
            )
    
    def addbatchpar(self, parval:list, parname:str, unit:str=None, desc:str=None):
        """Add new batch parameter.

        Args:
            parval (Iterable): list of parameter values.
            parname (str): name of the parameter.
            unit (str, optional): units of the parameter. Defaults to None.
            desc (str, optional): description of the parameter. Defaults to None.
        """        
        self._units.update({parname: unit})
        self._desc.update({parname: desc})
        self._pardict.update({parname: parval})

    def addbatchpars(
            self, parvals:Iterable[list], parnames:Iterable[str],
            units:dict|list[str]=None, descs:dict|list[str]=None
    ):
        """Add new batch parameters. 

        Args:
            parvals (`Iterable[list]`): list of parameters' values.
            parnames (`Iterable[str]`): llist of parameters' names.
            units (`dict | list[str]`, optional): list or dictionary with parameters'
            units. For dictionary keys should be parameter names and values units.
            Defaults to None.
            descs (`dict | list[str]`, optional): list or dictionary with parameters'
            descriptions. For dictionary keys should be parameter names and values
            units. Defaults to None.
        """        
        self._pardict = {
            par: i for par, i in zip(parnames,parvals)
        }
        if units is None:
            self._units.update(
                {k:None for k in parnames}
            )
        elif isinstance(units, abc.Mapping):
            self._units.update(units)
        else:
            self._units.update(
                {k:units[i] for i,k in enumerate(parnames)}
            )

        if descs is None:
            self._desc.update(
                {k:None for k in parnames}
            )
        elif isinstance(descs, abc.Mapping):
            self._desc.update(descs)
        else:
            self._desc.update(
                {k:descs[i] for i,k in enumerate(parnames)}
            )


    def __getattribute__(self, parnm:str):
        # if attribute does not exist for this object
        try:
            return object.__getattribute__(self, parnm)
        except AttributeError:
            pass
        # check dictionary with batch parameters
        try:
            return self._pardict[parnm]
        except KeyError:
            # if attribute is note returned by version of this method
            # from super class
            try:
                return super().__getattribute__(parnm)
            except KeyError:
                print(f"There is no atribute named {parnm}")       
            

    def makebatch(
            self,
            binds:Iterable[Iterable[str]]=None
    ) -> pd.DataFrame:           
        '''Prepares dataframe holding all parameters in batches (rows).
        Columns correspond to provided parameters. If provided, applies
        binds during batch creation. Each bind is a set of parameters
        that are dependent on a commom variable (if any of bound paramertes
        change between batches, then all bound parameters change).

        NOTE: the function creates dataframe with column order
        influenced by bindings. Therefore, order of columns may
        not match the order of provided parameters. To ensure
        desired order use features of the DataFrame class
        e.g. `df[['col1','col2','col3']]`.
        
        Args:
            binds (Iterable[Iterable[str], optional): iterable of binds.
            Each bind itself is an iterable of strings. Strings represent
            names of parameters.
            Defaults to None.

        Raises:
            ValueError: if any parameter name overlap between at lest two
            binds.

        Returns:
            DataFrame: dataframe with all batches in rows (columns correspond
            to the parameters).
            '''
        batches = None
        if binds is not None:
            # check if binds do not repeat
            bindsets = [set(b) for b in binds]
            if any([bool(el[0] & el[1]) for el in combinations(bindsets,2)]):
                print(binds)
                raise ValueError("Binds overlap!")
            
            # find keys of not bound parameters
            rest = [el if el not in chain(*binds) else None for el in self._pardict.keys()]
            rest = list(filterfalse(lambda item: not item, rest))
            
            for b in binds:
                if batches is not None:
                    batches = batches.merge(
                        pd.DataFrame.from_dict({k: self._pardict[k] for k in b})
                        ,how="cross"
                    )
                else:
                    batches = pd.DataFrame.from_dict({k: self._pardict[k] for k in b})
            
            for r in rest:
                batches = batches.merge(
                    pd.DataFrame.from_dict({k: self._pardict[k] for k in r})
                    ,how="cross"
                )
        
        else:
            for p in self._pardict.keys():
                if batches is not None:
                    batches = batches.merge(
                        pd.DataFrame.from_dict({k: self._pardict[k] for k in p})
                        ,how="cross"
                    )
                else:
                    batches = pd.DataFrame.from_dict({k: self._pardict[k] for k in p})
        self._batches = batches
        return self._batches
    

class Scalsource:
    # base class for source of radiation
    def __init__(
            self,
            amp:typing.ArrayLike,
            pha:typing.ArrayLike,
            wln:float
    ) -> None:
        self._amp = amp
        self._pha = pha
        self.wln = wln
    
    @property
    def amp(self):
        """Amplitude of the scalar wave"""
        return self._amp

    @amp.setter
    def amp(self, mat):
        self._amp = mat

    @amp.deleter
    def amp(self):
        del self._amp

    @property
    def pha(self):
        """Phase of the scalar wave"""
        return self._pha
    
    @pha.setter
    def pha(self, mat):
        self._pha=mat

    @pha.deleter
    def pha(self):
        del self._pha


class Scalpw(Scalsource):
    def __init__(
            self,
            shape:tuple[int, int] | int,
            wln:float,
            dtype=float
    ) -> None:
        if isinstance(shape, int):
            shape = (shape, shape)
        super().__init__(
            amp=np.ones(shape, dtype=dtype),
            pha=np.exp(1j * np.ones(shape) * 2*np.pi),
            wln=wln
        )


class Scalgaus(Scalsource):
    def __init__(
            self,
            shape:tuple[int, int] | int,
            samp:float,
            wln:float,
            w0:float,
            n:float=1.0,
            z:float=0.0,
            p:float=0.0
    ) -> None:
        if isinstance(shape, int):
            shape = (shape, shape)
        (x,y) = coord(shape,samp)
        self.w0 = w0
        self.n = n
        self.z = z
        self.p = p
        r = distcent(x,y)
        k = kk(wln)
        zr = w02zr(w0,wln,n)
        gb = gaussianbeamk(w0,z,zr,p,r,k)
        super().__init__(
            getAmp(gb),
            getPha(gb),
            wln
        )