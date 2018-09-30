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
        """Name the variable for pretty printing. Does not affect `eval_`"""
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
        return values[self]

    def ranged_eval(self, values, min=None, max=None, precondition=None):
        """Exclusive ranged evaluation, with optional precondition"""
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
        self_location = self.order(values)
        pre_self = self_location
        post_self = len(values) - 1 - self_location
        gradient_array = [0]*pre_self + [1] + [0]*post_self
        return np.array(gradient_array)

    @staticmethod
    def exp(var):
        if isinstance(var, Variable):
            return Variable(eval_=lambda values: math.e ** var.eval_(values),
                            grad=lambda values: (math.e ** var.eval_(values))*var.grad(values),
                            representation=lambda: "(e ** %s)" % str(var))
        if isinstance(var, (float, int)):
            return math.e ** var

    @staticmethod
    def log(var):
        if isinstance(var, Variable):
            return Variable(eval_=lambda values: math.log(var.ranged_eval(values, min=0, precondition=lambda k: k >= 0)),
                            # the precondition for eval_ deals with floating point rounding errors while still showing errors
                            grad=lambda values: (var.ranged_eval(values, min=0, precondition=lambda k: k >= 0) ** -1)*var.grad(values),
                            representation=lambda: "ln(%s)" % str(var))
        if isinstance(var, (float, int)):
            return math.log(var)

    def __add__(self, other):
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
        return self + (other * -1)

    def __rsub__(self, other):
        return self * -1 + other

    def __mul__(self, other):
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
        if isinstance(other, Variable):
            # logarithmic differentiation? TODO
            return NotImplemented
        if isinstance(other, (float, int)):
            return Variable(eval_=lambda values: self.eval_(values) ** other,
                            grad=lambda values: (other)*(self.eval_(values) ** (other - 1))*self.grad(values),
                            representation=lambda: "(%s ** %s)" % (str(self), str(other)))

    # __rpow__ not implemented; we simply don't have the rules for it

    def __truediv__(self, other):
        reciprocal = other ** -1
        return self * reciprocal

    def __rtruediv__(self, other):
        reciprocal = self ** -1
        return self * reciprocal

    def order(self, values):
        """Returns the order of the variable in the given list of values,
        as it would be returned by the gradient function, for example."""
        order = sorted([hash(key) for key in values.keys()])
        return order.index(hash(self))
