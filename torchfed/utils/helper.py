def interface_join(*args):
    """Join a list of interface names into a single interface name."""
    args_str = []
    for arg in args:
        if isinstance(arg, str):
            args_str.append(arg)
        elif callable(arg):
            args_str.append(arg.__name__)
        else:
            raise ValueError(f"{type(arg)} is not a valid interface name")
    return "/".join(args_str)
