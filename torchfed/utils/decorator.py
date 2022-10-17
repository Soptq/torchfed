from functools import wraps


def exposed(f):
    f.exposed = True

    @wraps(f)
    def wrapping(*args, **kwargs):
        result = f(*args, **kwargs)
        if result is None:
            raise Exception(f"Exposed function must have return values: {f}")
        return result

    return wrapping
