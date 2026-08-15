from climemu.utils import Registry

__version__ = "0.1.12"

"""
Registery of pretrained emulators for usage
"""
EMULATORS = Registry()


def build_emulator(name, frequency="monthly", **kwargs):
    model = EMULATORS[(name, frequency)](**kwargs)
    return model


from .emulators import Bouabid2026MonthlyEmulator, Bouabid2026DailyEmulator
__all__ = ['build_emulator', 'Bouabid2026MonthlyEmulator', 'Bouabid2026DailyEmulator']