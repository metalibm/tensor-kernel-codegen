import functools
import operator


from metalibm_core.core.ml_operations import (
    ML_Operation,
    GeneralArithmeticOperation,
    Variable, Addition, Multiplication,
    ControlFlowOperation,
    Constant,
    TableStore, TableLoad,
    Loop, Statement,
    ReferenceAssign, ML_LeafNode,
)

from metalibm_core.core.ml_formats import (
    ML_Int32, ML_Void,
)

from metalibm_core.core.advanced_operations import PlaceHolder


class Tensor:
    """ Tensor object """
    def __init__(self, base_buffer, tensor_descriptor):
        self.base_buffer = base_buffer
        self.descriptor = tensor_descriptor

class TensorDescriptor:
    """ Tensor parameters descriptor """
    def __init__(self, sdim, strides, scalar_format):
        """
            :arg sdim: number of elements in each dimension
            :arg strides: stride between 2 elements along one axis / dimension (in # of elements)

            stride can either be a Constant or a Variable
        """
        self.sdim = sdim
        self.strides = strides
        self.scalar_format = scalar_format

    def __str__(self):
        return "T({})[{}]".format(" x ".join("{}(s={})".format(dim, stride) for dim, stride in zip(self.sdim, self.strides)), str(self.scalar_format))

    def get_bounding_size(self):
        """ return the total number of element in the minimal linearized array 
            containing the tensor """
        bounding_size = self.get_linear_index_from_multi([dim - 1 for dim in self.sdim]) + 1
        print("{}'s bounding size is {}".format(str(self),bounding_size))
        return bounding_size
        #strided_dims = [dim*stride for dim, stride in zip(self.sdim, self.strides)]
        #return functools.reduce(operator.mul, strided_dims, 1)

    def get_multi_index_from_linear(self, linear_index):
        """ Transform a linear index into a multi-dimension index """
        sub_index_list = []
        #for dim_size, stride in zip(self.sdim, self.strides):
        for stride, stride_p1 in zip(self.strides, self.strides[1:] + [None]):
            if stride_p1 is None:
                sub_index = linear_index // stride
            else:
                sub_index = (linear_index // stride) % stride_p1
            assert sub_index < self.sdim[len(sub_index_list)]
            sub_index_list.append(sub_index)
        return sub_index_list
    def get_linear_index_from_multi(self, multi_index):
        """ Transform a multi-dimension index in a linarized one """
        return self.generate_linearized_offset(multi_index)

    def generate_linearized_offset(self, sub_index_list):
        """ Generate the offset to access tensor element located
            at sub_index_list """
        extended_index_list = [(self.strides[i] * sub_index) for i, sub_index in enumerate(sub_index_list)] 
        return functools.reduce(operator.add, extended_index_list)

class Accessor(ML_Operation):
    """ common accessor: Read/Write operation to a Tensor """
    def __init__(self, tensor, index_expr):
        self.tensor = tensor
        self.index_expr = index_expr

class ReadAccessor(Accessor):
    """ Read operation from a Tensor """
    def __init__(self, tensor, index_expr, value_format):
        Accessor.__init__(self, tensor, index_expr)
        self.value_format = value_format
class WriteAccessor(Accessor):
    """ Write operation to a Tensor """
    def __init__(self, tensor, index_expr, value_expr):
        Accessor.__init__(self, tensor, index_expr)
        self.value_expr = value_expr

class Range(ML_Operation):
    def __init__(self, first_index, last_index, index_step=1):
        self.first_index = first_index
        self.last_index = last_index
        self.index_step = index_step

class IterRange(Range):
    """ Iterator range with explicit iteration variable """
    def __init__(self, var_index, first_index, last_index, index_step=1):
        Range.__init__(self, first_index, last_index, index_step)
        self.var_index = var_index

class Sum(GeneralArithmeticOperation):
    """ Compound summation of arbitrary length """
    arity = 2
    def __init__(self, elt_operation, index_iter_range, **kw):
        super().__init__(elt_operation, index_iter_range, **kw)
        # self.elt_operation = elt_operation
        # self.index_iter_range = index_iter_range

    @property
    def elt_operation(self):
        return self.get_input(0)
    @property
    def index_iter_range(self):
        return self.get_input(1)

class NDRange:
    """ high-level N-dimensionnal range kernel description """
    def __init__(self, var_range_list, kernel):
        # kernel is executed on var_range_list
        self.var_range_list = var_range_list
        self.kernel = kernel

def expand_kernel_expr(kernel, iterator_format=ML_Int32):
    """ Expand a kernel expression into the corresponding MDL graph """
    if isinstance(kernel, Sum):
        var_iter = kernel.index_iter_range.var_index
        # TODO/FIXME to be uniquified
        acc = Variable("acc", var_type=Variable.Local, precision=kernel.precision)
        # TODO/FIXME implement proper acc init
        scheme = Loop(
            Statement(
                ReferenceAssign(var_iter, kernel.index_iter_range.first_index),
                ReferenceAssign(acc, Constant(0, precision=kernel.precision))
            ),
            var_iter <= kernel.index_iter_range.last_index,
            Statement(
                ReferenceAssign(acc, acc + expand_kernel_expr(kernel.elt_operation)),
                # loop iterator increment
                ReferenceAssign(var_iter, var_iter + kernel.index_iter_range.index_step)
            )
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
    """ Expand an accessor node into a valid MDL description """
    if isinstance(accessor, ReadAccessor):
        # check dimensionnality: the number of sub-indexes in ReadAccessor's
        # index_expr must match the dimensionnality of ReadAccessor's tensor
        # tensor_descriptor
        return TableLoad(accessor.tensor.base_buffer, accessor.tensor.descriptor.generate_linearized_offset(accessor.index_expr), precision=accessor.value_format)
    elif isinstance(accessor, WriteAccessor):
        return TableStore(
            expand_kernel_expr(accessor.value_expr),
            accessor.tensor.base_buffer,
            accessor.tensor.descriptor.generate_linearized_offset(accessor.index_expr),
            precision=ML_Void,
        )
    else:
        raise NotImplementedError

def extract_placeholder(node, memoization_map=None):
    """ assunming node is an operation expression (not a Statement)
        containing PlaceHolder nodes, this function extract the PlaceHolder
        nodes, remove them for the expression and return them in a list by order
        of appearance, alongside the placeholder-free expression

        :arg node: root of the operation graph to process
        :type node: ML_Operation
        :return: pair(node, statement_list), placeholder-free node and
                 extracted statement
    """
    if memoization_map is None:
        memoization_map = {}

    if node in memoization_map:
        return memoization_map[node]
    else:
        result = None
        if isinstance(node, PlaceHolder):
            head = node.get_input(0)
            statement_list = []
            for op in node.inputs[1:]:
                _, local_list = extract_placeholder(op, memoization_map)
                statement_list = statement_list + local_list
            # process head after internal PlaceHolder statement
            head, head_list = extract_placeholder(head, memoization_map)
            statement_list = statement_list + head_list
            # returning head for node value (de-capsulating PlaceHolder)
            result = head, statement_list
        elif isinstance(node, (Statement, Loop, ControlFlowOperation)):
            return None, [node]
        elif isinstance(node, Statement):
            statement_list = []
            for op in node.inputs:
                new_op, local_list = extract_placeholder(op, memoization_map)
                if not new_op is None:
                    statement_list = statement_list + [new_op]
                statement_list = statement_list + local_list
            result = None, statement_list
        elif isinstance(node, ML_LeafNode):
            result = node, []
        else:
            statement_list = []
            for index, op in enumerate(node.inputs):
                new_node, local_list = extract_placeholder(op, memoization_map)
                statement_list = statement_list + local_list
                node.set_input(index, new_node)
            result = node, statement_list
        # memoizing result
        memoization_map[node] = result
        return result


def expand_ndrange(ndrange):
    """ Expand an ndrange object into a MDL graph """
    def expand_sub_ndrange(var_range_list, kernel):
        if len(var_range_list) == 0:
            pre_expanded_kernel = expand_kernel_expr(kernel)
            expanded_kernel, statement_list = extract_placeholder(pre_expanded_kernel)
            return Statement(*tuple(statement_list), expanded_kernel)
        else:
            var_range = var_range_list.pop(0)
            scheme = Loop(
                # init statement
                ReferenceAssign(var_range.var_index, var_range.first_index),
                # exit condition
                var_range.var_index <= var_range.last_index,
                # loop body
                Statement(
                    expand_sub_ndrange(var_range_list, kernel),
                    # loop iterator increment
                    ReferenceAssign(var_range.var_index, var_range.var_index + var_range.index_step)
                ),
            )
        return scheme
    return expand_sub_ndrange(ndrange.var_range_list, ndrange.kernel)


