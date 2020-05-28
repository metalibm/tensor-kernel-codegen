from sollya import Interval


from metalibm_core.core.ml_complex_formats import ML_Pointer_Format
from metalibm_core.core.ml_formats import (ML_Binary32, ML_Int32, ML_Void)
from metalibm_core.core.ml_operations import (
    Variable, Multiplication, Statement, Return, Constant)


from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_core.utility.ml_template import (
    DefaultArgTemplate, ML_NewArgTemplate)

from meta_tensor_function import MetaTensorFunction

from tensor_iterator import (
    Tensor,
    TensorDescriptor,
    NDRange, WriteAccessor, ReadAccessor,
    IterRange, Sum,

    expand_ndrange, tile_ndrange, exchange_loop_order
)

class ShallowConvKernel(MetaTensorFunction):
    """ Meta shallow convolution kernel, with reduced
        number of channels"""
    function_name = "shallow_conv_kernel"
    arity = 6

    def __init__(self, args=DefaultArgTemplate):
        # list of argument indexes (in generated function argument list)
        # corresponding to output (resp. input) tensors
        output_tensor_indexes = [0]
        input_tensor_indexes = [1, 2]
        MetaTensorFunction.__init__(self, output_tensor_indexes, input_tensor_indexes, args)
        self.kernel_size = args.kernel_size
        # patch output format
        self.implementation.set_output_format(ML_Void)

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for ML_Exponential,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_mmk = {
            "output_file": "shallow_conv_kernel.c",
            "function_name": "shallow_conv_kernel",
            "test_index_range": [[16, 32], [16, 32], [16, 32]],
            "auto_test_range": [Interval(-1, 1), Interval(-1, 1)],
            "precision": ML_Binary32,
            "target": GenericProcessor.get_target_instance()
        }
        default_args_mmk.update(kw)
        return DefaultArgTemplate(**default_args_mmk)


    def generate_scheme(self):
        size_format = ML_Int32

        # Matrix storage
        in_storage = self.implementation.add_input_variable("buffer_in", ML_Pointer_Format(self.precision))
        kernel_storage = self.implementation.add_input_variable("buffer_kernel", ML_Pointer_Format(self.precision))
        out_storage = self.implementation.add_input_variable("buffer_out", ML_Pointer_Format(self.precision))

        # Matrix sizes
        w = self.implementation.add_input_variable("w", size_format)
        h = self.implementation.add_input_variable("h", size_format)


        # A is a (n x p) matrix in row-major
        tIn = Tensor(in_storage, TensorDescriptor([w, h], [1, w], self.precision))
        # B is a (p x m) matrix in row-major
        kernel_strides = [1]
        for previous_dim in self.kernel_size[:-1]:
            kernel_strides.append(previous_dim * kernel_strides[-1])
        print("kernel_strides: {}".format(kernel_strides))
        tKernel = Tensor(kernel_storage, TensorDescriptor(self.kernel_size, kernel_strides, self.precision))
        # C is a (n x m) matrix in row-major
        tOut = Tensor(out_storage, TensorDescriptor([w, h], [1, w], self.precision))

        index_format = ML_Int32

        # main NDRange description
        i = Variable("i", precision=index_format, var_type=Variable.Local)
        j = Variable("j", precision=index_format, var_type=Variable.Local)
        k_w = Variable("k_w", precision=index_format, var_type=Variable.Local)
        k_h = Variable("k_h", precision=index_format, var_type=Variable.Local)
        result = NDRange(
            [IterRange(i, 0, w-1), IterRange(j, 0, h -1)],
            WriteAccessor(
                tOut, [i, j],
                Sum(
                    Sum(Multiplication(
                        ReadAccessor(tIn, [i + k_w, j - k_h], self.precision),
                        ReadAccessor(tKernel, [k_w, k_h], self.precision)),
                        IterRange(k_w, -(self.kernel_size[0]-1) // 2, (self.kernel_size[0]-1) // 2),
                        precision=self.precision
                    ),
                    IterRange(k_h, -(self.kernel_size[1]-1) // 2, (self.kernel_size[1]-1) // 2),
                    precision=self.precision)))

        mdl_scheme = expand_ndrange(result)
        print("mdl_scheme:\n{}".format(mdl_scheme.get_str(depth=None)))
        return Statement(
            mdl_scheme,
            Return()
        )

    def get_ordered_arg_tuple(self, tensor_descriptors, input_tables, output_tables):
        (input_tensor_descriptor_list, output_tensor_descriptor_list) = tensor_descriptors

        tA_desc = input_tensor_descriptor_list[0]
        tB_desc = input_tensor_descriptor_list[1]
        p = tA_desc.sdim[0]
        n = tA_desc.sdim[1]
        m = tB_desc.sdim[0]

        index_format = ML_Int32

        return (
            input_tables[0], input_tables[1],
            output_tables[0],
            Constant(n, precision=index_format),
            Constant(m, precision=index_format),
            Constant(p, precision=index_format),
        )

    def tensor_element_emulate(self, tensor_descriptors, output_tensor_id, linear_id, input_tables):
        # matrix kernel only expects a single output tensor
        assert output_tensor_id == 0

        (input_tensor_descriptor_list, output_tensor_descriptor_list) = tensor_descriptors
        out_nd_index = output_tensor_descriptor_list[0].get_multi_index_from_linear(linear_id)

        j, i = out_nd_index
        acc = 0
        tA_desc = input_tensor_descriptor_list[0]
        tB_desc = input_tensor_descriptor_list[1]
        p = tA_desc.sdim[0]
        assert p == tB_desc.sdim[1]
        for k in range(p):
            index_A = tA_desc.get_linear_index_from_multi([k, i])
            index_B = tB_desc.get_linear_index_from_multi([j, k])
            acc += input_tables[0][index_A] * input_tables[1][index_B]
        return acc

    def generate_output_tensor_descriptors(self, random_sizes):
        """ generate list of instance of output tensor descriptors for testing """
        n, m, p = random_sizes
        # C is a (n x m) matrix in row-major
        tC_desc = TensorDescriptor([m, n], [1, m], self.precision)
        return [tC_desc]
    def generate_innput_tensor_descriptors(self, random_sizes):
        """ generate list of instance of input tensor descriptors for testing """
        n, m, p = random_sizes
        tA_desc = TensorDescriptor([p, n], [1, p], self.precision)
        # B is a (p x m) matrix in row-major
        tB_desc = TensorDescriptor([m, p], [1, m], self.precision)
        return [tA_desc, tB_desc]


if __name__ == "__main__":
    arg_template = ML_NewArgTemplate(default_arg=ShallowConvKernel.get_default_args())
    # extra arguments
    arg_template.get_parser().add_argument(
        "--test-index-range", dest="test_index_range", default=[[16, 32], [16, 32], [16, 32]],
        type=eval,
        action="store", help="random range for matrix sizes")
    arg_template.get_parser().add_argument(
        "--kernel-size", dest="kernel_size", default=(3,3),
        type=(lambda s: tuple(map(int, s.split(",")))),
        action="store", help="random range for matrix sizes")
    # argument extraction
    args = arg_template.arg_extraction()

    shallow_conv_kernel = ShallowConvKernel(args)

    shallow_conv_kernel.gen_implementation()
