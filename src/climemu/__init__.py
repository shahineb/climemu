from climemu.utils import Registry

__version__ = "0.1.9"

"""
Registery of pretrained emulators for usage
"""
EMULATORS = Registry()


def build_emulator(name, **kwargs):
    model = EMULATORS[name](**kwargs)
    return model


from .emulators import Bouabid2025Emulator
__all__ = ['build_emulator', 'Bouabid2025Emulator']