from abc import ABC, abstractmethod


class AbstractEmulator(ABC):    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Generate climate data samples"""
        pass

    @classmethod
    def build(cls, **kwargs):
        return cls(**kwargs)


class GriddedEmulator(AbstractEmulator):
    @property
    @abstractmethod
    def lat(self):
        """Latitude coordinates"""
        pass
    
    @property
    @abstractmethod
    def lon(self):
        """Longitude coordinates"""
        pass
    
    @property
    @abstractmethod
    def vars(self):
        """Variable list"""
        pass

    @property
    def nlat(self):
        return len(self.lat)
    
    @property
    def nlon(self):
        return len(self.lon)
    
    @property
    def nvar(self):
        return len(self.vars)