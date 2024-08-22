def optimize(x, f):
    diff = 10
    h = 0.01
    while diff > 0.01:
        f_prime = derivative(x, f, h)
        f_double_prime = second_derivative(x, f, h)
        x_temp = x - f_prime/f_double_prime
        diff = abs(x_temp - x)
        x = x_temp
    return x, f(x)

def derivative(x, f, h):
    return (f(x+h)-f(x))/h

def second_derivative(x, f, h):
    return (derivative(x+h, f, h)-derivative(x, f, h))/h