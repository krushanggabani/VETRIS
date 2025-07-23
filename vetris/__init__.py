_initialized = False
_backend = None

def init(theme="dark", eps=1e-15, backend=None):
    global _initialized, _backend
    if _initialized:
        raise RuntimeError("Env already initialized.")
    
    _backend = backend
    _initialized = True
    
    
    return {
        "theme": theme,
        "eps": eps,
        "backend": _backend
    }

def raise_exception(msg="Something went wrong."):
    # helper to centralize your errors
    raise RuntimeError(msg)


if __name__ == "__main__":

    cfg = init(theme="dark", backend="cpu")
    print("Initialized:", cfg)
