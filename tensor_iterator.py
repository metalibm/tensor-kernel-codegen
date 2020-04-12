import functools

from metalibm_core.core.ml_operations import (
    ML_Operation,
    Variable, Addition, Multiplication,
    TableStore, TableLoad,
    Loop, Statement,
    ReferenceAssign, ML_LeafNode,
)

from metalibm_core.core.advanced_operations import PlaceHolder


class Tensor:
    def __init__(self, base_buffer, tensor_descriptor):
        self.base_buffer = base_buffer
        self.descriptor = tensor_descriptor

class TensorDescriptor:
    def __init__(self, sdim, strides):
        """
            :arg sdim: number of elements in each dimension
            :arg strides: stride between 2 elements along one axis / dimension

            stride can either be a Constant or a Variable
        """
        self.sdim = sdim
        self.strides = strides

    def generate_linearized_offset(self, sub_index_list):
        """ Generate the offset to access tensor element located
            at sub_index_list """
        extended_index_list = [Multiplication(self.sdim[i], sub_index) for i, sub_index in enumerate(sub_index_list)] 
        return functools.reduce(Addition, extended_index_list)

class Accessor(ML_Operation):
    def __init__(self, tensor, index_expr):
        self.tensor = tensor
        self.index_expr = index_expr

class ReadAccessor(Accessor):
    pass
class WriteAccessor(Accessor):
    def __init__(self, tensor, index_expr, value_expr):
        Accessor.__init__(self, tensor, index_expr)
        self.value_expr = value_expr

class Range:
    def __init__(self, first_index, last_index, index_step=None):
        self.first_index = first_index
        self.last_index = last_index
        self.index_step = index_step

class IterRange(Range):
    def __init__(self, var_index, first_index, last_index, index_step=None):
        Range.__init__(self, first_index, last_index, index_step)
        self.var_index = var_index

class Sum:
    def __init__(self, elt_operation, index_iter_range):
        self.elt_operation = elt_operation
        self.index_iter_range = index_iter_range

class NDRange:
    def __init__(self, var_range_list, kernel):
        # kernel is executed on var_range_list
        self.var_range_list = var_range_list
        self.kernel = kernel

def expand_kernel_expr(kernel):
    """ Expand a kernel expression into the corresponding MDL graph """
    if isinstance(kernel, Sum):
        var_iter = kernel.index_iter_range.var_index
        # TODO/FIXME to be uniquified
        acc = Variable("acc", var_type=Variable.Local)
        # TODO/FIXME implement proper acc init
        scheme = Loop(
            Statement(
                ReferenceAssign(var_iter, kernel.index_iter_range.first_index),
                ReferenceAssign(acc, 0)
            ),
            var_iter <= kernel.index_iter_range.last_index,
            expand_kernel_expr(kernel.elt_operation)
        )
        return PlaceHolder(acc, scheme)
    elif isinstance(kernel, (ReadAccessor, WriteAccessor)):
        return expand_accessor(kernel)
    elif isinstance(kernel, ML_LeafNode):
        return kernel
    else:
        # vanilla metalibm ops are left unmodified (except
        # recursive expansion)
        for index, op in enumerate(kernel.inputs):
            new_op = expand_kernel_expr(op)
            kernel.set_input(index, new_op)
        return kernel

def expand_accessor(accessor):
    if isinstance(accessor, ReadAccessor):
        # check dimensionnality: the number of sub-indexes in ReadAccessor's
        # index_expr must match the dimensionnality of ReadAccessor's tensor
        # tensor_descriptor
        return TableLoad(accessor.tensor.base_buffer, accessor.tensor.descriptor.generate_linearized_offset(accessor.index_expr))
    elif isinstance(accessor, WriteAccessor):
        return TableStore(
            accessor.tensor.base_buffer,
            accessor.tensor.descriptor.generate_linearized_offset(accessor.index_expr),
            expand_kernel_expr(accessor.value_expr))
    else:
        raise NotImplementedError

def expand_ndrange(ndrange):
    """ Expand an ndrange object into a MDL graph """
    def expand_sub_ndrange(var_range_list, kernel):
        if len(var_range_list) == 0:
            return expand_kernel_expr(kernel)
        else:
            var, var_range = var_range_list.pop(0)
            scheme = Loop(
                # init statement
                ReferenceAssign(var, var_range.first_index),
                # exit condition
                var <= var_range.last_index,
                # loop body
                expand_sub_ndrange(var_range_list, kernel)
            )
        return scheme
    return expand_sub_ndrange(ndrange.var_range_list, ndrange.kernel)



if __name__ == "__main__":
    # matrix multiply
    n = Variable("n")
    m = Variable("m")
    p = Variable("p")
    A_storage = Variable("buffer_a")
    B_storage = Variable("buffer_b")
    C_storage = Variable("buffer_c")
    # A is a (n x p) matrix in row-major
    tA = Tensor(A_storage, TensorDescriptor([p, n], [1, p]))
    # B is a (p x m) matrix in row-major
    tB = Tensor(B_storage, TensorDescriptor([m, p], [1, m]))
    # C is a (n x m) matrix in row-major
    tC = Tensor(C_storage, TensorDescriptor([m, n], [1, m]))

    #
    i = Variable("i")
    j = Variable("j")
    k = Variable("k")
    result = NDRange([(i, Range(0, n -1)), [j, Range(0, m-1)]], WriteAccessor(tC, [i, j], Sum(Multiplication(ReadAccessor(tA, [i, k]), ReadAccessor(tB, [k, j])), IterRange(k, 0, p - 1))))

    mdl_scheme = expand_ndrange(result)
    print("mdl_scheme:\n{}".format(mdl_scheme.get_str(depth=None)))


    # Convolutions
