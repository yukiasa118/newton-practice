import numdifftools as nd
import scipy
def multi_optimize(x, f):
    """
    Multivariate Newton's Method

    Return the coordinates of the optimal point found using Newton's Method.

    Parameters
    -----------
    x: array of int or float, initial guess for Newton's method
    f: function, the function on which Newton's method is performed

    Returns
    ----------
    x: input value to get the optimal point found using Newton's Method
    f(x): the optimized value for the function
    
    """
    if not callable(f):
        raise TypeError(f"Argument is not a function, it is of type {type(f)}")

    
    iter_limit = 10000000000000
    iter_count = 0
    diff = 10
    h = 0.00001
    while diff > 0.00001 and iter_count < iter_limit:
        grad = nd.Gradient(f)
        hess = nd.Hessian(f)

        x_temp = x - scipy.linalg.solve(hess(x), grad(x))
        diff = abs(x_temp - x)
        x = x_temp
        iter_count = iter_count + 1
    return [x, f(x)]

