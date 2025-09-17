import types


class Registry(dict):
    """
    A helper class for managing access to builders, it extends a dictionary
    and provides a registering functions than can be used as a decorator

    Creating a registry:
        MODULES = Registry()

    There two types of builder callable you can register:

    (1) : Functions which can be registered by either a simple call
        ```
        def build_bar():
            return True
        MODULES.register('bar', build_bar)
        ```
        or using a decorator at function definition
        ```
        @MODULES.register('bar')
        def build_bar():
            return True
        ```

    (2) : Class constructor method cls.build which again can be registered
    with a call if class has a cls.build method
        ```
        foo = Foo()
        MODULES.register('Foo', foo)
        ```
        of using a decorator at class definition
        ```
        @MODULES.register('foo')
        class Foo:
            @classmethod
            def build(cls, *args, **kwargs):
                # build a class instance foo
                return foo
        ```

    Access of module is just like using a dictionary, eg:
        build_bar = MODULES['bar']
        build_foo = MODULES['foo']
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