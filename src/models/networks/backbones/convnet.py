from abc import ABC, abstractmethod
import jax.numpy as jnp
import equinox as eqx


class ConvNet(eqx.Module, ABC):
    _input_size: tuple
    _output_size: tuple
    _base_kwargs = dict()

    def __init__(self, input_size):
        """
        Args:
            input_size (tuple[int]): e.g. (C, H, W) for a 2D input
        """
        self._input_size = input_size
        self._output_size = None

    def _init_kwargs_path(self, conv_kwargs, nb_filters):
        """Initializes convolutional path making sure making sure it
        matches the number of filters dimensions

        Returns a list of kwargs for each convolutional layer as :

        [{'kernel_size': 3, 'bn': True, 'stride': 2},
            {'kernel_size': 3, 'bn': True, 'stride': 2},
            {'kernel_size': 3, 'bn': False, 'stride': 1, 'padding': 1}]

        This convolutional path length must match the number of filters specified.
        If a single dictionnary is provided, it is replicated to match the length
        of the number of filters.

        Args:
            conv_kwargs (dict, list[dict]): convolutional block kwargs
            nb_filters (list[int]): number of filter of each block

        Returs:
            type: list[dict]
        """
        conv_kwargs = {} if conv_kwargs is None else conv_kwargs
        if isinstance(conv_kwargs, list):
            assert len(conv_kwargs) == len(nb_filters), "Kwargs and number of filters length must match"
            return [{**self._base_kwargs, **kwargs} for kwargs in conv_kwargs]
        elif isinstance(conv_kwargs, dict):
            return len(nb_filters) * [{**self._base_kwargs, **conv_kwargs}]
        else:
            raise TypeError("kwargs must be of type dict or list[dict]")

    def _hidden_dimension_numel(self):
        """Computes number of elements of hidden dimension
        """
        raise NotImplementedError

    def _compute_output_size(self, *args, **kwargs):
        """
        Runs a dummy forward pass to determine the output shape.
        """
        dummy_input = jnp.zeros(self._input_size)
        output = self.__call__(dummy_input, *args, **kwargs)
        return output.shape

    @abstractmethod
    def __call__(self, x, *args, **kwargs):
        """
        Args:
            x (torch.Tensor): (N, C, W, H)
        """
        pass

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        if self._output_size is None:
            object.__setattr__(self, '_output_size', self._compute_output_size())
        return self._output_size

    @input_size.setter
    def input_size(self, input_size):
        self._input_size = input_size

    @output_size.setter
    def output_size(self, output_size):
        self._output_size = output_size