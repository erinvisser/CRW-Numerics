def default_two_walker_kernels():
    # Walker 1: up/down rare, left/right common
    K1 = {"up": 1/8, "down": 1/8, "left": 3/8, "right": 3/8}
    # Walker 2: flipped directions
    K2 = {"up": 3/8, "down": 3/8, "left": 1/8, "right": 1/8}
    return [K1, K2]

def validate_kernel(K: dict) -> None:
    s = K["up"] + K["down"] + K["left"] + K["right"]
    if abs(s - 1.0) > 1e-12:
        raise ValueError(f"Kernel probabilities must sum to 1, got {s}")