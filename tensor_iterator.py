import functools
import operator


from metalibm_core.core.ml_operations import (
    ML_Operation,
    ML_ArithmeticOperation,
    Variable, Addition, Multiplication,
    ControlFlowOperation,
    Constant,
    TableStore, TableLoad,
    Loop, Statement,
    ReferenceAssign,
    is_leaf_node,
    VectorAssembling,
)

from metalibm_core.core.ml_formats import (
    ML_Int32, ML_Void,
)

from metalibm_core.core.advanced_operations import PlaceHolder
from metalibm_core.core.legalizer import extract_placeholder
from metalibm_core.core.ml_vectorizer import vectorize_format

from metalibm_core.utility.log_report import Log


class Tensor:
    """ Tensor object """
    def __init__(self, base_buffer, tensor_descriptor):
        self.base_buffer = base_buffer
        self.descriptor = tensor_descriptor

    def __str__(self):
        return "Tensor <{}> from {}".format(self.descriptor, self.base_buffer)

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
        return "T({})[{}]".format(" x ".join("{}(s={})".format(str(dim).replace("\n",""), str(stride).replace("\n","")) for dim, stride in zip(self.sdim, self.strides)), str(self.scalar_format))

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

    def get_affine_factor(self, accessor, vectorized_index):
        """ determine what is the affine factor when incremeting
            vectorized_index in accessor to access self tensor """ 
        affine_factor = None
        for i, sub_index in enumerate(accessor.index_expr):
            # TODO/FIXME: support expressions
            assert isinstance(sub_index, Variable)
            if sub_index == vectorized_index:
                if affine_factor is None:
                    affine_factor = self.strides[i]
                else:
                    # already encountered variable
                    affine_factor += self.strides[i]
        return affine_factor



class Accessor(ML_Operation):
    """ common accessor: Read/Write operation to a Tensor """
    # number of contiguous elements accessed (starting at <index_expr>)
    num_celt = None
    def __init__(self, tensor, index_expr):
        self.tensor = tensor
        self.index_expr = index_expr
    def get_str(self, *args, **kw):
        return str(self)
    def __str__(self):
        return "{} access of\n    {}\n  with indexes: {}\n (num_celt={})".format(self.__class__.__name__, self.tensor, ", ".join(index.get_str() for index in self.index_expr), self.num_celt)

    def get_dimension_index(accessor, vectorized_index):
        """ return a list of all dimension index which depends on vectorized_index """
        dimension_index = []
        for i, sub_index in enumerate(accessor.index_expr):
            # TODO/FIXME: support expressions
            assert isinstance(sub_index, Variable)
            if sub_index == vectorized_index:
                dimension_index.append(i)
        return dimension_index

class ReadAccessor(Accessor):
    """ Read operation from a Tensor """
    # default accessor is 1-element
    num_celt = 1
    def __init__(self, tensor, index_expr, value_format):
        Accessor.__init__(self, tensor, index_expr)
        self.value_format = value_format
class WriteAccessor(Accessor):
    """ Write operation to a Tensor """
    # default accessor is 1-element
    num_celt = 1
    def __init__(self, tensor, index_expr, value_expr):
        Accessor.__init__(self, tensor, index_expr)
        self.value_expr = value_expr

class ReadVectorAccessor(ReadAccessor):
    def __init__(self, tensor, index_expr, value_format, num_celt=1, vector_dim=None):
        ReadAccessor.__init__(self, tensor, index_expr, value_format)
        self.num_celt = num_celt
        # index of the dimensions along which the vector access is made
        # Notes: only vector access along a single dimension are supported
        self.vector_dim = vector_dim

class WriteVectorAccessor(WriteAccessor):
    def __init__(self, tensor, index_expr, value_expr, num_celt=1, vector_dim=None):
        WriteAccessor.__init__(self, tensor, index_expr, value_expr)
        self.num_celt = num_celt
        # index of the dimensions along which the vector access is made
        # Notes: only vector access along a single dimension are supported
        self.vector_dim = vector_dim

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

    def __str__(self):
        return "IterRange({})[{}:{}:{}]".format(self.var_index, self.first_index, self.last_index, self.index_step)
    def get_str(self, *args, **kw):
        return str(self)

class Sum(ML_ArithmeticOperation):
    """ Compound summation of arbitrary length """
    arity = 2
    name = "Sum"
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
    if isinstance(kernel, NDRange):
        return expand_ndrange(kernel)
    elif isinstance(kernel, Sum):
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
    elif is_leaf_node(kernel):
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


def substitute_var(node, var_map, memoization_map=None):
    """ process operation graph starting from node, 
        and change any node in var_map by var_map[node].var_index """
    if memoization_map is None:
        memoization_map = {}
    if node in memoization_map:
        return memoization_map[node]
    elif node in var_map:
        result = var_map[node].var_index
    elif isinstance(node, ReadAccessor):
        node.index_expr = [substitute_var(sub_index, var_map, memoization_map) for sub_index in node.index_expr]
        result = node
    elif isinstance(node, WriteAccessor):
        node.index_expr = [substitute_var(sub_index, var_map, memoization_map) for sub_index in node.index_expr]
        node.value_expr = substitute_var(node.value_expr, var_map, memoization_map)
        result = node
    elif isinstance(node, IterRange): 
        node.var_index = substitute_var(node.var_index, var_map, memoization_map)
        node.first_index = substitute_var(node.first_index, var_map, memoization_map)
        node.last_index = substitute_var(node.last_index, var_map, memoization_map)
        node.index_step = substitute_var(node.index_step, var_map, memoization_map)

        result = node
    elif isinstance(node, int):
        # FIXME: maybe int index should be wrapper as Constant
        return node
    elif is_leaf_node(node):
        result = node
    else:
        for index, op in enumerate(node.inputs):
            new_op = substitute_var(op, var_map, memoization_map)
            node.set_input(index, new_op)
        result = node
    memoization_map[node] = result
    return result


def tile_ndrange(ndrange, tile, index_format=ML_Int32):
    """ inplace transform ndrange such that it iterate over a sub-tile of
        size tile rather than a single element
        tile is a dict(var_index -> tile_dim) """
    # The transformation is performed by replacing each range
    # implicating one of the variable from tile, by a range whose step is the tile's dimension
    # and then adding a sub-iterange using a sub-alias for the tile's variable whose range
    # is [0; tile's dimension - 1]
    new_var_range_list = []
    var_transform_map = {}
    kernel_var_range_list = []
    # transform var_range_list
    for iter_range in ndrange.var_range_list:
        var_index = iter_range.var_index
        if var_index in tile:
            tile_dim = tile[var_index]
            new_iter_range = IterRange(var_index, iter_range.first_index, iter_range.last_index, index_step=tile_dim)
            new_var_range_list.append(new_iter_range)
            sub_var = Variable("sub_%s" % var_index.get_tag(), precision=index_format, var_type=Variable.Local)
            sub_var_range = IterRange(sub_var, var_index, var_index + tile_dim-1)
            kernel_var_range_list.append(sub_var_range)
            var_transform_map[iter_range.var_index] = sub_var_range
        else:
            new_var_range_list.append(iter_range)
    # tile kernel
    new_kernel = substitute_var(ndrange.kernel, var_transform_map)
    sub_ndrange = NDRange(kernel_var_range_list, new_kernel)
    return NDRange(new_var_range_list, sub_ndrange)


def offset_read_accessor(accessor, index_offset_map):
    """ create a new read accessor from accessor, while offseting
        each index based of index_offset_map (dict of expr -> offset) """
    def substitude_in_expr(expr, index_offset_map):
        if expr in index_offset_map:
            offset = index_offset_map[expr]
            if offset == 0:
                return expr
            else:
                return expr + offset
        elif is_leaf_node(expr):
            return expr
        else:
            # shallow copy
            expr = copy.copy(expr)
            for op_index, op in enumerate(expr.inputs):
                expr.set_input(op_index, substitude_in_expr(op, index_offset_map))
            return expr

    offseted_index_expr = [substitude_in_expr(index_op, index_offset_map) for index_op in accessor.index_expr]
    new_accessor = ReadAccessor(accessor.tensor, offseted_index_expr, accessor.value_format)
    return new_accessor

class VectorBroadcast(VectorAssembling):
    """ Specialization of VectorAssembling operator which broadcast a single
        element to all the vector lanes """
    arity = 1
    name = "VectorBroadcast"

def vectorize_read_accessor(accessor, vectorized_index, vector_size):
    """ Vectorize an Accessor  by expanding it alongside <index_to_size>'s key Index
        of an amount equal to <index_to_size>' value """
    assert isinstance(accessor, ReadAccessor)
    affine_factor = accessor.tensor.descriptor.get_affine_factor(accessor, vectorized_index)
    vector_format = vectorize_format(accessor.value_format, vector_size)
    if affine_factor == 1:
        vector_dim_index = accessor.get_dimension_index(vectorized_index)
        assert len(vector_dim_index) == 1
        # vectorizable access
        vector_accessor = ReadVectorAccessor(
            accessor.tensor,
            accessor.index_expr,
            vector_format,
            num_celt=vector_size,
            vector_dim=vector_dim_index[0])
        return vector_accessor
    elif affine_factor is None:
        # independent access => Broadcast
        return VectorBroadcast(accessor, precision=vector_format)
    else:
        # gather
        element_tuple = tuple(offset_read_accessor(accessor, {vectorized_index: offset}) for offset in range(vector_size))
        return VectorAssembling(*element_tuple, precision=vector_format)


def vectorize_write_accessor(write_accessor, vectorized_index, vector_size):
    assert isinstance(write_accessor, WriteAccessor)
    assert write_accessor.num_celt == 1
    affine_factor = write_accessor.tensor.descriptor.get_affine_factor(write_accessor, vectorized_index)
    if affine_factor != 1:
        Log.report(Log.Error, "vectorize_kernel only supports root WriteAccessor with affine_factor=1 (not {})", affine_factor)

    vector_dim_index = write_accessor.get_dimension_index(vectorized_index)
    assert len(vector_dim_index) == 1

    vector_write_accessor = WriteVectorAccessor(
        write_accessor.tensor,
        write_accessor.index_expr,
        vectorize_kernel_value(write_accessor.value_expr, vectorized_index, vector_size),
        num_celt=vector_size,
        vector_dim=vector_dim_index[0])

    return vector_write_accessor

def kernel_depends_on_index(kernel, index_var):
    """ test whether the expression / operation graph kernel depends
        on index_var or not """
    if kernel == index_var:
        return True
    elif isinstance(kernel, int):
        return False
    elif isinstance(kernel, ReadAccessor):
        return any(kernel_depends_on_index(index, index_var) for index in kernel.index_expr)
    elif isinstance(kernel, WriteAccessor):
        return any(kernel_depends_on_index(index, index_var) for index in kernel.index_expr) or \
               kernel_depends_on_index(kernel.value_expr, index_var)
    elif is_leaf_node(kernel):
        # not index_var (excluded by first test)
        return False
    else:
        return any(kernel_depends_on_index(op, index_var) for op in kernel.inputs)

def iter_range_depends_on_index(iter_range, index_var):
    assert isinstance(iter_range, IterRange)
    return (
        kernel_depends_on_index(iter_range.first_index, index_var) or \
        kernel_depends_on_index(iter_range.last_index, index_var) or \
        kernel_depends_on_index(iter_range.var_index, index_var))

def vectorize_kernel_value(kernel, vectorized_index, vector_size):
    if isinstance(kernel, ReadAccessor):
        vectorized_kernel  = vectorize_read_accessor(kernel, vectorized_index, vector_size)
        return vectorized_kernel
    elif isinstance(kernel, WriteAccessor):
        vectorized_kernel = vectorize_write_accessor(kernel, vectorized_index, vector_size)
        return vectorized_kernel
    elif isinstance(kernel, IterRange):
        if not iter_range_depends_on_index(kernel, vectorized_index):
            return kernel
        else:
            # case when IterRange depends on the vectorization index is not supported
            raise NotImplementedError
    else:
        if not kernel_depends_on_index(kernel, vectorized_index):
            vectorized_kernel = VectorBroadcast(kernel, precision=vectorize_format(kernel.get_precision(), vector_size))
        elif kernel == vectorized_index:
            element_tuple = tuple(vectorized_index + offset for offset in range(vector_size))
            vector_format = vectorize_format(kernel.get_precision(), vector_size)
            return VectorAssembling(*element_tuple, precision=vector_format)
        elif isinstance(kernel, ML_ArithmeticOperation):
            vectorized_kernel = kernel.copy(copy_map={op: vectorize_kernel_value(op, vectorized_index, vector_size) for op in kernel.inputs})
            vectorized_kernel.set_precision(vectorize_format(kernel.get_precision(), vector_size))
            return vectorized_kernel
        else:
            print("unsupported kernel in vectorize_kernel_value: {}\n".format(kernel))
            raise NotImplementedError 
    return vectorized_kernel



def exchange_loop_order(ndrange, new_order):
    """ inplace modification the iteration order in ndrange by re-ordering
        iter_range index according to new_order (list of indexes) """
    assert len(new_order) == len(ndrange.var_range_list)
    new_var_range_list = [ndrange.var_range_list[index] for index in new_order]
    ndrange.var_range_list = new_var_range_list
    return ndrange


def expand_ndrange(ndrange):
    """ Expand an ndrange object into a MDL graph """
    def expand_sub_ndrange(var_range_list, kernel):
        if len(var_range_list) == 0:
            pre_expanded_kernel = expand_kernel_expr(kernel)
            expanded_kernel, statement_list = extract_placeholder(pre_expanded_kernel)
            expanded_statement = Statement(*tuple(statement_list))
            print("expand_ndrange: ", expanded_kernel, statement_list)
            if not expanded_kernel is None:
                # append expanded_kernel at the Statement's end once
                # every PlaceHolder's dependency has been resolved
                expanded_statement.add(expanded_kernel)
            return expanded_statement
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


