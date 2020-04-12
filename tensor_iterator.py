from metalibm_core.core.ml_operations import (
    ML_Operation,
    Variable, Addition, Multiplication)


class TensorIterator:
    def __init__(self, sdim, strides):
        """
            :arg sdim: number of dimensions
            :arg strides: stride between 2 element along one axis / dimension

            stride can either be a Constant or a Variable
        """
        self.sdim = sdim
        self.strides = strides


class Accessor(ML_Operation):
    def __init__(self, input_buffer, index_expr):
        self.input_buffer = input_buffer
        self.index_expr = index_expr

class ReadAccessor(Accessor):
    pass
class WriteAccessor(Accessor):
    def __init__(self, output_buffer, index_expr, value_expr):
        Accessor.__init__(self, output_buffer, index_expr)
        self.value_expr = value_expr

class Range:
    def __init__(self, first_index, last_index, index_step=None):
        self.first_index = first_index
        self.last_index = last_index
        self.index_step = index_step

class Sum:
    def __init__(self, elt_operation, index_range):
        self.elt_operation = elt_operation
        self.index_range = index_range

class NDRange:
    def __init__(self, var_range_list, kernel):
        # kernel is executed on var_range_list
        self.var_range_list = var_range_list
        self.kernel = kernel


if __name__ == "__main__":
    # matrix multiply
    n = Variable("n")
    m = Variable("m")
    p = Variable("p")
    # A is a (n x p) matrix in row-major
    tA = TensorIterator([p, n], [1, p])
    # B is a (p x m) matrix in row-major
    tB = TensorIterator([m, p], [1, m])
    # C is a (n x m) matrix in row-major
    tC = TensorIterator([m, n], [1, m])

    #
    i = Variable("i")
    j = Variable("j")
    k = Variable("k")
    result = NDRange([(i, Range(0, n -1)), [j, Range(0, m-1)]], WriteAccessor(tC, [i, j], Sum(Multiplication(ReadAccessor(tA, [i, k]), ReadAccessor(tB, [k, j])), Range(k, 0, p - 1))))


    # Convolutions
