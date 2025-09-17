"""Tests for the Registry class in climemu.utils.registry."""

import pytest
from src.climemu.utils.registry import Registry


class TestRegistry:
    """Test cases for the Registry class."""

    def test_registry_initialization(self):
        """Test that Registry initializes as a dictionary."""
        registry = Registry()
        assert isinstance(registry, dict)
        assert len(registry) == 0

    def test_register_function(self):
        """Test registering a function."""
        registry = Registry()
        
        def test_func():
            return "test_result"
        
        registry.register('test_func', test_func)
        assert registry['test_func'] == test_func
        assert registry['test_func']() == "test_result"

    def test_register_function_decorator(self):
        """Test registering a function using decorator."""
        registry = Registry()
        
        @registry.register('decorated_func')
        def decorated_function():
            return "decorated_result"
        
        assert registry['decorated_func']() == "decorated_result"

    def test_register_class(self):
        """Test registering a class with build method."""
        registry = Registry()
        
        class TestClass:
            @classmethod
            def build(cls, **kwargs):
                return cls(**kwargs)
            
            def __init__(self, value="default"):
                self.value = value
        
        registry.register('test_class', TestClass)
        instance = registry['test_class'](value="custom")
        assert isinstance(instance, TestClass)
        assert instance.value == "custom"

    def test_duplicate_registration_raises_error(self):
        """Test that duplicate registration raises error."""
        registry = Registry()
        
        def func1():
            return "first"
        
        def func2():
            return "second"
        
        registry.register('duplicate', func1)
        
        with pytest.raises(AssertionError):
            registry.register('duplicate', func2)

    def test_register_unknown_type_raises_error(self):
        """Test that registering unknown type raises error."""
        registry = Registry()
        
        with pytest.raises(TypeError):
            registry.register('unknown', "not_a_function_or_class")
