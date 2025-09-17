from src.climemu.utils import Registry

"""
Registery of pretrained emulators for usage
"""
EMULATORS = Registry()


def build_emulator(name):
    model = EMULATORS[name]()
    return model


from .emulators import Bouabid2025Emulator
__all__ = ['build_emulator', 'Bouabid2025Emulator']