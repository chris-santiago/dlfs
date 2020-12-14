from numpy import ndarray


def assert_same_shape(array_1: ndarray, array_2: ndarray):
    """Ensure proper shapes."""
    msg = f"""
        Two ndarrays should have the same shape;
        instead, first ndarray's shape is {tuple(array_1.shape)}
        and second ndarray's shape is {tuple(array_2.shape)}.
        """
    assert array_1.shape == array_2.shape, msg
    return None
