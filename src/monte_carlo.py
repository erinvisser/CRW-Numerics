import random

# k = 0 is Walker 1: up/down rare, left/right common
# K1 = {"up": 1/8, "down": 1/8, "left": 3/8, "right": 3/8}
# k = 1 is Walker 2: flipped directions
# K2 = {"up": 3/8, "down": 3/8, "left": 1/8, "right": 1/8}

def random_step(i: int, j: int, kernel: int, ):
    """
    Updates the current position by flipping three coins and 
    taking a step according to the walker choice.

    Parameters
    ----------
    i : int
        current row index
    j : int
        current column index
    kernel : int
        choice of walker; 0 = K1, 1 = K2

    Returns
    -------
    int
        updated row index
    int
        updated column index
    """
    flips = [random.choice(["H", "T"]) for _ in range(3)]
    nT = flips.count("T")
    if kernel == 0: # K1
        if nT == 3:
            i += 1
        elif nT == 2:
            j += 1
        elif nT == 1:
            j -= 1
        elif nT == 0:
            i -= 1
        else:
            raise ValueError("Number of tails must be between 0 and 3.")
    elif kernel == 1: # K2
        if nT == 3:
            j -= 1
        elif nT == 2:
            i += 1
        elif nT == 1:
            i -= 1
        elif nT == 0:
            j += 1
        else:
            raise ValueError("Number of tails must be between 0 and 3.")
    else:
        raise ValueError("Kernel choice must be 'K1' or 'K2'.")

    return i, j

def run_simulation(
    domain
    policy
    starting point
):

1. verify interior domain and policy of same dimension
2. save boundary 
3. set starting point
loop:
- update counter
1. check policy at point
2. run random_step 
3. check if at boundary

if at boundary, exit loop and compute payout

    return counter, payout, trajectory, movie ?

def check_convergence(
    starting point
    u at that point
    number of trials
):

run_simulation N times and compute average payout
compare average payout with u value

def 