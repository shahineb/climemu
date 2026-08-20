from earthsampler.utils import Registry

__version__ = "0.2.0"

"""
Registery of pretrained emulators for usage
"""
EMULATORS = Registry()


def build_emulator(name, frequency="monthly", **kwargs):
    """Create an emulator instance for a given Earth System Model.

    Args:
        name: ESM identifier (e.g. ``"MPI-ESM1-2-LR"``, ``"MIROC6"``).
        frequency: Temporal frequency — ``"monthly"`` or ``"daily"``.
        **kwargs: Forwarded to the emulator constructor. Notably
            ``variables`` (list of str) to subset output variables.

    Returns:
        An emulator instance. Call ``.load()`` then ``.compile(n_samples)``
        before generating samples.

    Raises:
        KeyError: If the ``(name, frequency)`` pair is not registered.

    Example:
        >>> emulator = build_emulator("MPI-ESM1-2-LR")
        >>> emulator.load()
        >>> emulator.compile(n_samples=5)
        >>> samples = emulator(gmst=2.0, month=6, xarray=True)
    """
    model = EMULATORS[(name, frequency)](**kwargs)
    return model


from .emulators import Bouabid2026MonthlyEmulator, Bouabid2026DailyEmulator
__all__ = ['build_emulator', 'Bouabid2026MonthlyEmulator', 'Bouabid2026DailyEmulator']