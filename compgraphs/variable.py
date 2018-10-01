"""
This file defines the Variable, a class used for basic mathematical operations
and gradient calculations.

Authors: MILES MCCAIN and LIV MARTENS
License: GPLv3
"""


import random
import numpy as np
import math

class Variable():
    def __init__(self, eval_=None, grad=None, representation=None, name=None):
        self.identifier = hash(random.random())
        if eval_ != None:
            self.eval_ = eval_
        if grad != None:
            self.grad = grad
        self.representation = representation
        self.name = name

    def name(self, name):
        """Name the variable for pretty printing. Doesn't affect `eval_`.
        
        Arguments:
            name {string} -- the name of the variable
        """
        self.name = name

    def __repr__(self):
        if self.representation != None:
            return self.representation()
        if self.name != None:
            return self.name
        return "<%s>" % str(hash(self))[:3]

    def __hash__(self):
        return self.identifier

    def eval_(self, values):
        """Evaluate the variable with the given values.
        
        Arguments:
            values {dictionary} -- a dictionary of values, where the
            keys are variable objects and their values are floats
        
        Returns:
            float -- the value of the evaluated variable
        """
        return values[self]

    def ranged_eval(self, values, min=None, max=None, precondition=None):
        """Perform a ranged evaluation of the variable on the given values.
        This method allows a range to be specified as well as a precondition,
        which helps to prevent floating point rounding errors.
        
        Arguments:
            values {dictionary} -- a dictionary of the values (see `eval_`)
        
        Keyword Arguments:
            min {float} -- the miniumum output value (not enforced if None) (default: {None})
            max {float} -- the maxiumum output value (not enforced if None) (default: {None})
            precondition {function} -- a function that returns true or false on the pre-ranged value (default: {None})
        
        Raises:
            Exception -- if the precondition fails
        
        Returns:
            float -- the evaluated value
        """
        value = self.eval_(values)
        if precondition != None:
            if not precondition(value):
                raise Exception("precondition not met (value=%s)" % value)
        if min != None and value <= min:
            value = np.nextafter(min, min + 1)
        if max != None and value >= max:
            value = np.nextafter(max, max - 1)
        return value

    def grad(self, values):
        """Calculate the gradient at any given set of values.
        
        Arguments:
            values {dictionary} -- the values of the variable
            (only include the required values!)
        
        Returns:
            np.array -- the gradient vector
        """

        self_location = self.order(values)
        pre_self = self_location
        post_self = len(values) - 1 - self_location
        gradient_array = [0]*pre_self + [1] + [0]*post_self
        return np.array(gradient_array)

    @staticmethod
    def exp(var):
        """Exponentiate the variable (e ** var)
        
        Arguments:
            var {Variable, float, int} -- the variable to exponentiate
        
        Returns:
            Variable -- a variable that has been exponentiated
        """

        if isinstance(var, Variable):
            return Variable(eval_=lambda values: math.e ** var.eval_(values),
                            grad=lambda values: (math.e ** var.eval_(values))*var.grad(values),
                            representation=lambda: "(e ** %s)" % str(var))
        if isinstance(var, (float, int)):
            return math.e ** var

    @staticmethod
    def log(var):
        """Take the logarithm (base e) of the given variable.
        
        Arguments:
            var {Variable, float, int} -- the variable to take the logarithm of
        
        Returns:
            Variable -- a variable that represents the log of the given variable
        """

        if isinstance(var, Variable):
            return Variable(eval_=lambda values: math.log(var.ranged_eval(values, min=0, precondition=lambda k: k >= 0)),
                            # the precondition for eval_ deals with floating point rounding errors while still showing errors
                            grad=lambda values: (var.ranged_eval(values, min=0, precondition=lambda k: k >= 0) ** -1)*var.grad(values),
                            representation=lambda: "ln(%s)" % str(var))
        if isinstance(var, (float, int)):
            return math.log(var)

    def __add__(self, other):
        """Add two variables together.
        
        Arguments:
            other {Variable, float, int} -- the variable to add
        
        Returns:
            Variable -- the added variables
        """

        if isinstance(other, Variable):
            return Variable(eval_=lambda values: self.eval_(values) + other.eval_(values),
                            grad=lambda values: self.grad(values) + other.grad(values),
                            representation=lambda: "(%s + %s)" % (str(self), str(other)))
        if isinstance(other, (float, int)):
            return Variable(eval_=lambda values: self.eval_(values) + other,
                            grad=lambda values: self.grad(values),
                            representation=lambda: "(%s + %s)" % (str(self), str(other)))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        """Subtract a variable from another.
        
        Arguments:
            other {Variable, float, int} -- the variable to subtract
        
        Returns:
            Variable -- the result of the subtraction operation
        """

        return self + (other * -1)

    def __rsub__(self, other):
        return self * -1 + other

    def __mul__(self, other):
        """Multiply two variables together.
        
        Arguments:
            other {Variable, float, int} -- the variable to multiply by
        
        Returns:
            Variable -- the result of the multiplication operation
        """

        if isinstance(other, Variable):
            return Variable(eval_=lambda values: self.eval_(values) * other.eval_(values),
                            grad=lambda values: self.grad(values)*other.eval_(values) + other.grad(values)*self.eval_(values),
                            representation=lambda: "(%s * %s)" % (str(self), str(other)))
        if isinstance(other, (float, int)):
            return Variable(eval_=lambda values: self.eval_(values) * other,
                            grad=lambda values: self.grad(values) * other,
                            representation=lambda: "(%s * %s)" % (str(self), str(other)))
    
    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        """Raise a variable to the power of a _constant_.
        
        Arguments:
            other {float, int} -- the value to raise the variable to
        
        Returns:
            Variable -- the variable raised to the given power
        """

        if isinstance(other, Variable):
            # logarithmic differentiation? TODO
            return NotImplemented
        if isinstance(other, (float, int)):
            return Variable(eval_=lambda values: self.eval_(values) ** other,
                            grad=lambda values: (other)*(self.eval_(values) ** (other - 1))*self.grad(values),
                            representation=lambda: "(%s ** %s)" % (str(self), str(other)))

    # __rpow__ not implemented; we simply don't have the rules for it

    def __truediv__(self, other):
        """Divide a variable by another.
        
        Arguments:
            other {Variable, float, int} -- the denominator value/variable
        
        Returns:
            Variable -- a variable of the result of the division operation
        """

        reciprocal = other ** -1
        return self * reciprocal

    def __rtruediv__(self, other):
        reciprocal = self ** -1
        return self * reciprocal

    def order(self, values):
        """Returns the order of the variable in the given list of values,
        as it would be returned by the gradient function, for example.
        
        Arguments:
            values {dictionary} -- the dictionary of values
        
        Returns:
            int -- the index of the variable within the dictionary (and in,
            for example, a gradient vector)
        """

        order = sorted([hash(key) for key in values.keys()])
        return order.index(hash(self))
