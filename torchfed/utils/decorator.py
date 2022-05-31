from functools import wraps


def exposed(f):
    f.exposed = True

    @wraps(f)
    def wrapping(*args, **kwargs):
        result = f(*args, **kwargs)
        if result is None:
            raise Exception("Exposed function must have return values")
        return result

    return wrapping
