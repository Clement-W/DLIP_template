import importlib
from omegaconf import DictConfig


def instanciate_class(class_str, *args, **kwargs):
    """
    Instantiates a class from a string.

    Args:
        class_str (str): The name of the class to instantiate. 
                         It should be in the format 'module.ClassName'.
        *args: The arguments to pass to the class's constructor.
        **kwargs: The keyword arguments to pass to the class's constructor.

    Returns:
        object: An instance of the class specified by `class_str`.
    """
    cls = get_class(class_str)

    # Check if the last argument is a DictConfig and use it as kwargs
    if args and isinstance(args[-1], DictConfig):
        kwargs.update(args[-1])
        args = args[:-1]

    return cls(*args, **kwargs)

    # return instance


def get_class(class_str):
    """
    Returns a class from a string.

    Args:
        class_str (str): The name of the class to return. 
                         It should be in the format 'module.ClassName'.

    Returns:
        class: The class corresponding to `class_str`.
    """
    # Split the string into module and class names
    module_name, class_name = class_str.rsplit('.', 1)

    # Import the module
    module = importlib.import_module(module_name)

    # Get the class and return it
    cls = getattr(module, class_name)
    return cls
