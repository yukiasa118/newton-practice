def optimize(x, f):
    """
    Newton's Method

    Return the coordinates of the optimal point found using Newton's Method.

    Parameters
    -----------
    x: int or float, initial guess for Newton's method
    f: function, the function on which Newton's method is performed

    Returns
    ----------
    x: input value to get the optimal point found using Newton's Method
    f(x): the optimized value for the function
    
    """
    if not callable(f):
        raise TypeError(f"Argument is not a function, it is of type {type(f)}")
    if not isinstance(x, float) and not isinstance(x, int):
        raise TypeError('x must be numeric')
    
    iter_limit = 10000000000
    iter_count = 0
    diff = 10
    h = 0.00001
    while diff > 0.00001 and iter_count < iter_limit:
        f_prime = derivative(x, f, h)
        f_double_prime = second_derivative(x, f, h)
        if f_double_prime == 0:
            raise RunTimeError("second derivative is 0")
        x_temp = x - f_prime / f_double_prime
        diff = abs(x_temp - x)
        x = x_temp
        iter_count = iter_count + 1
    return [x, f(x)]


def derivative(x, f, h):
    return (f(x + h) - f(x)) / h


def second_derivative(x, f, h):
    return (derivative(x + h, f, h) - derivative(x, f, h)) / h