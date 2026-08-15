from climemu.utils import Registry

__version__ = "0.1.12"

"""
Registery of pretrained emulators for usage
"""
EMULATORS = Registry()


def build_emulator(name, **kwargs):
    model = EMULATORS[name](**kwargs)
    return model


from .emulators import Bouabid2026Emulator
__all__ = ['build_emulator', 'Bouabid2026Emulator']