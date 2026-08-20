import types


class Registry(dict):
    """Dictionary-based registry for builder callables.

    Supports registration via function call or decorator. Access registered
    builders like a regular dictionary.

    Example::

        MODULES = Registry()

        @MODULES.register('bar')
        def build_bar():
            return True

        build_bar = MODULES['bar']
    """

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, name, module_object=None):
        # Used as a decorator
        if module_object is None:
            def register_func(module_object):
                self.register(name=name, module_object=module_object)
                return module_object
            return register_func

        # Used as a function call
        else:
            if isinstance(module_object, type):
                self._register_generic(module_dict=self, name=name, builder=module_object.build)
            elif isinstance(module_object, types.FunctionType):
                self._register_generic(module_dict=self, name=name, builder=module_object)
            else:
                raise TypeError("Trying to register unknown data type")

    @staticmethod
    def _register_generic(module_dict, name, builder):
        assert name not in module_dict
        module_dict[name] = builder