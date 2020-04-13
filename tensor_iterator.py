import functools

from metalibm_core.core.ml_operations import (
    ML_Operation,
    GeneralArithmeticOperation,
    Variable, Addition, Multiplication,
    ControlFlowOperation,
    Constant, Return,
    TableStore, TableLoad,
    Loop, Statement,
    ReferenceAssign, ML_LeafNode,
)

from metalibm_core.core.ml_formats import (
    ML_Binary32, ML_Int32, ML_Void,
)
from metalibm_core.core.ml_complex_formats import ML_Pointer_Format

from metalibm_core.core.advanced_operations import PlaceHolder
from metalibm_core.core.ml_function import ML_FunctionBasis

from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_core.utility.ml_template import (
    DefaultArgTemplate, ML_NewArgTemplate)


class Tensor:
    """ Tensor object """
    def __init__(self, base_buffer, tensor_descriptor):
        self.base_buffer = base_buffer
        self.descriptor = tensor_descriptor

class TensorDescriptor:
    """ Tensor parameters descriptor """
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
    def __init__(self, first_index, last_index, index_step=None):
        self.first_index = first_index
        self.last_index = last_index
        self.index_step = index_step

class IterRange(Range):
    """ Iterator range with explicit iteration variable """
    def __init__(self, var_index, first_index, last_index, index_step=None):
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
            ReferenceAssign(acc, acc + expand_kernel_expr(kernel.elt_operation))
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


class MatrixMultiplyKernel(ML_FunctionBasis):
    """ Meta matrix multiply kernel """
    function_name = "matrix_multiply_kernel"
    arity = 6

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for ML_Exponential,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_mmk = {
            "output_file": "mm_kernel.c",
            "function_name": "mm_kernel",
            "precision": ML_Binary32,
            "target": GenericProcessor.get_target_instance()
        }
        default_args_mmk.update(kw)
        return DefaultArgTemplate(**default_args_mmk)


    def generate_scheme(self):
        size_format = ML_Int32

        # Matrix sizes
        n = self.implementation.add_input_variable("n", size_format)
        m = self.implementation.add_input_variable("m", size_format)
        p = self.implementation.add_input_variable("p", size_format)

        # Matrix storage
        A_storage = self.implementation.add_input_variable("buffer_a", ML_Pointer_Format(self.precision))
        B_storage = self.implementation.add_input_variable("buffer_b", ML_Pointer_Format(self.precision))
        C_storage = self.implementation.add_input_variable("buffer_c", ML_Pointer_Format(self.precision))

        # A is a (n x p) matrix in row-major
        tA = Tensor(A_storage, TensorDescriptor([p, n], [1, p]))
        # B is a (p x m) matrix in row-major
        tB = Tensor(B_storage, TensorDescriptor([m, p], [1, m]))
        # C is a (n x m) matrix in row-major
        tC = Tensor(C_storage, TensorDescriptor([m, n], [1, m]))

        index_format = ML_Int32

        #
        i = Variable("i", precision=index_format)
        j = Variable("j", precision=index_format)
        k = Variable("k", precision=index_format)
        result = NDRange(
            [(i, Range(0, n -1)), [j, Range(0, m-1)]],
            WriteAccessor(
                tC, [i, j],
                Sum(
                    Multiplication(
                        ReadAccessor(tA, [i, k], self.precision),
                        ReadAccessor(tB, [k, j], self.precision)),
                    IterRange(k, 0, p - 1),
                    precision=self.precision)))

        mdl_scheme = expand_ndrange(result)
        print("mdl_scheme:\n{}".format(mdl_scheme.get_str(depth=None)))
        return Statement(
            mdl_scheme,
            Return()
        )


def example():
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
    # TODO


if __name__ == "__main__":
    arg_template = ML_NewArgTemplate(default_arg=MatrixMultiplyKernel.get_default_args())
    # argument extraction
    args = arg_template.arg_extraction()

    matrix_multiply_kernel = MatrixMultiplyKernel(args)

    matrix_multiply_kernel.gen_implementation()
