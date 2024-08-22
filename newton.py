def optimize(x, f):
    """
    Newton's Method

    returns the coordinates of the optimal point found using Newton's Method.

    Parameters
    -----------
    x: int or float, initial guess for Newton's method
    f: function, the function on which Newton's method is performed

    Returns
    ----------
    x: input value to get the optimal point found using Newton's Method
    f(x): the optimized value for the function
    
    """
    iter_limit = 100000000
    iter_count = 0
    diff = 10
    h = 0.01
    while diff > 0.01 and iter_count < iter_limit:
        f_prime = derivative(x, f, h)
        f_double_prime = second_derivative(x, f, h)
        x_temp = x - f_prime / f_double_prime
        diff = abs(x_temp - x)
        x = x_temp
        iter_count = iter_count + 1
    return x, f(x)


def derivative(x, f, h):
    return (f(x + h) - f(x)) / h


def second_derivative(x, f, h):
    return (derivative(x + h, f, h) - derivative(x, f, h)) / h
