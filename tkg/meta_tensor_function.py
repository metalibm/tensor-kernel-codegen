import random

import sollya
from sollya import Interval, inf, sup

from metalibm_core.core.ml_formats import ML_Int32, ML_UInt32, ML_Void, ML_Int64
from metalibm_core.core.array_function import (
    ML_ArrayFunction,
    generate_1d_table, generate_2d_table
)
from metalibm_core.core.random_gen import get_precision_rng
from metalibm_core.core.special_values import FP_QNaN

from metalibm_core.utility.ml_template import DefaultArgTemplate
from metalibm_core.core.ml_table import ML_NewTable

from metalibm_core.code_generation.code_function import (
    CodeFunction, FunctionGroup
)
from metalibm_core.code_generation.generator_utility import (
    FunctionOperator, TemplateOperatorFormat,
    FO_Arg,
)
from metalibm_core.core.ml_operations import (
    Variable, Constant,
    Loop, ReferenceAssign, Statement,
    TableLoad,
    FunctionObject,
    Return, ConditionBlock,
    Addition,
)

class MetaTensorFunction(ML_ArrayFunction):
    def __init__(self,
                 output_tensor_args_indexes,
                 input_tensor_args_indexes,
                 args=DefaultArgTemplate):
        ML_ArrayFunction.__init__(self, args)
        self.output_tensor_args_indexes = output_tensor_args_indexes
        self.input_tensor_args_indexes = input_tensor_args_indexes

    def generate_output_tensor_descriptors(self, random_sizes):
        """ generate list of instance of output tensor descriptors for testing """
        raise NotImplementedError
    def generate_innput_tensor_descriptors(self, random_sizes):
        """ generate list of instance of input tensor descriptors for testing """
        raise NotImplementedError

    def generate_random_sizes(self):
        return [random.randrange(lo, hi+1) for lo,hi in self.test_index_range]

    def generate_test_tables(self, test_num, test_ranges=[Interval(-1.0, 1.0)]):
        """ Generate inputs and output table to be shared between auto test
            and max_error tests """
        random_sizes = self.generate_random_sizes()
        output_tensor_descriptor_list = self.generate_output_tensor_descriptors(random_sizes)
        input_tensor_descriptor_list = self.generate_innput_tensor_descriptors(random_sizes)

        index_range = self.test_index_range

        # number of arrays expected as inputs for tested_function
        NUM_INPUT_ARRAY = len(input_tensor_descriptor_list)
        # position of the input array in tested_function operands (generally
        # equals to 1 as to 0-th input is often the destination array)
        INPUT_INDEX_OFFSET = 1


        # concatenating standard test array at the beginning of randomly
        # generated array
        INPUT_ARRAY_SIZE = [td.get_bounding_size() for td in input_tensor_descriptor_list]

        # TODO/FIXME: implement proper input range depending on input index
        # assuming a single input array
        input_precisions = [td.scalar_format for td in input_tensor_descriptor_list]
        rng_map = [get_precision_rng(precision, inf(test_range), sup(test_range)) for precision, test_range in zip(input_precisions, test_ranges)]

        # generated table of inputs
        input_tables = [
            generate_1d_table(
                INPUT_ARRAY_SIZE[table_id],
                input_precisions[table_id],
                self.uniquify_name("input_table_arg%d" % table_id),
                value_gen=(lambda _: input_precisions[table_id].round_sollya_object(rng_map[table_id].get_new_value(), sollya.RN))
            ) for table_id in range(NUM_INPUT_ARRAY)
        ]

        OUTPUT_ARRAY_SIZE = [td.get_bounding_size() for td in output_tensor_descriptor_list]
        OUTPUT_PRECISION = [td.scalar_format for td in output_tensor_descriptor_list]
        NUM_OUTPUT_ARRAY = len(output_tensor_descriptor_list)

        # generate output_array
        output_tables = [generate_1d_table(
            OUTPUT_ARRAY_SIZE[table_id],
            OUTPUT_PRECISION[table_id],
            self.uniquify_name("output_array_%d" % table_id),
            const=False,
            #value_gen=(lambda _: FP_QNaN(self.precision))
            value_gen=(lambda _: 0)
        ) for table_id in range(NUM_OUTPUT_ARRAY)]
        tensor_descriptors = (input_tensor_descriptor_list, output_tensor_descriptor_list)
        return tensor_descriptors, input_tables, output_tables

    def generate_test_wrapper(self, tensor_descriptors, input_tables, output_tables):
        auto_test = CodeFunction("test_wrapper", output_format = ML_Int32)

        tested_function    = self.implementation.get_function_object()
        function_name      = self.implementation.get_name()

        failure_report_op       = FunctionOperator("report_failure")
        failure_report_function = FunctionObject("report_failure", [], ML_Void, failure_report_op)

        printf_success_op = FunctionOperator("printf", arg_map = {0: "\"test successful %s\\n\"" % function_name}, void_function = True, require_header=["stdio.h"]) 
        printf_success_function = FunctionObject("printf", [], ML_Void, printf_success_op)

        # accumulate element number
        acc_num = Variable("acc_num", precision=ML_Int64, var_type=Variable.Local)

        test_loop = self.get_tensor_test_wrapper(
            tested_function,
            tensor_descriptors,
            input_tables,
            output_tables,
            acc_num,
            self.generate_tensor_check_loop)

        # common test scheme between scalar and vector functions
        test_scheme = Statement(
          test_loop,
          printf_success_function(),
          Return(Constant(0, precision = ML_Int32))
        )
        auto_test.set_scheme(test_scheme)
        return FunctionGroup([auto_test])


    def get_ordered_arg_tuple(self, tensor_descriptors, input_tables, output_tables):
        """ generate an ordered tuple of argument for the current function
            assuming 
            - tensor_descriptors is (input_tensor_descriptor_list, output_tensor_descriptor_list) 
            - input tensors are listed in input_tables,
            - output tensors are listed in output_tables
        """
        # must be overloaded by actual function
        raise NotImplementedError

    def get_tensor_test_wrapper(
            self,
            tested_function,
            tensor_descriptors,
            input_tables, output_tables,
            acc_num,
            post_statement_generator,
            NUM_INPUT_ARRAY=1):
        """ generate a test loop for multi-array tests
             @param test_num number of elementary array tests to be executed
             @param tested_function FunctionObject to be tested
             @param table_size_offset_array ML_NewTable object containing
                    (table-size, offset) pairs for multi-array testing
             @param input_table ML_NewTable containing multi-array test inputs
             @param output_table ML_NewTable containing multi-array test outputs
             @param post_statement_generator is generator used to generate
                    a statement executed at the end of the test of one of the
                    arrays of the multi-test. It expects 6 arguments:
                    (input_tables, output_array, table_size_offset_array,
                     array_offset, array_len, test_id)
             @param printf_function FunctionObject to print error case
        """
        array_len = Variable("len", precision=ML_UInt32, var_type=Variable.Local)


        def pointer_add(table_addr, offset):
            pointer_format = table_addr.get_precision_as_pointer_format()
            return Addition(table_addr, offset, precision=pointer_format)

        array_inputs    = tuple(input_tables[in_id] for in_id in range(NUM_INPUT_ARRAY))
        function_call = tested_function(
            *(self.get_ordered_arg_tuple(tensor_descriptors, input_tables, output_tables)))

        post_statement = post_statement_generator(
                            tensor_descriptors,
                            input_tables, output_tables)

        test_statement = Statement(
            function_call,
            post_statement,
        )

        return test_statement


    def get_printf_error_detail_fct(self, tensor_descriptor):
        output_format = tensor_descriptor.scalar_format
        # result is the second argument of the function (after erroenous element index)
        result_arg_id = 1
        # build the format string for result/expected display
        result_display_format = output_format.get_display_format().format_string
        result_display_vars = output_format.get_display_format().pre_process_fct("{%d}" % result_arg_id)

        template = ("printf(\"error[%u]: {fct_name},"
                    " result is {result_display_format} "
                    "vs expected \""
                    ", {{0}}, {result_display_vars}"
                    ")").format(
                        fct_name=self.function_name,
                        result_display_format=result_display_format,
                        result_display_vars=result_display_vars,
                    )
        printf_op = TemplateOperatorFormat(template, void_function=True, arity=(1+1), require_header=["stdio.h"]) 
        printf_error_detail_function = FunctionObject("printf", [ML_UInt32] + [output_format], ML_Void, printf_op)
        return printf_error_detail_function

    def generate_tensor_check_loop(self, tensor_descriptors, input_tables, output_tables):
        # unpack tensor descriptors tuple
        (input_tensor_descriptor_list, output_tensor_descriptor_list) = tensor_descriptors
        # internal array iterator index
        vj = Variable("j", precision=ML_UInt32, var_type=Variable.Local)

        printf_error_detail_function = self.get_printf_error_detail_fct(output_tensor_descriptor_list[0])

        NUM_INPUT_ARRAY = len(input_tables)

        # generate the expected table for the whole multi-array
        expected_tables = self.generate_expected_table(tensor_descriptors, input_tables)

        # global statement to list all checks
        check_statement = Statement()

        # implement check for each output tensor
        for out_id, out_td in enumerate(output_tensor_descriptor_list):
            # expected values for the (vj)-th entry of the sub-array
            expected_values = [TableLoad(expected_tables[out_id], vj, i) for i in range(self.accuracy.get_num_output_value())]
            # local result for the (vj)-th entry of the sub-array
            local_result = TableLoad(output_tables[out_id], vj)

            array_len = out_td.get_bounding_size()

            if self.break_error:
                return_statement_break = Statement(
                    printf_error_detail_function(*((vj,) + (local_result,))),
                    self.accuracy.get_output_print_call(self.function_name, output_values)
                )
            else:
                return_statement_break = Statement(
                    printf_error_detail_function(*((vj,) + (local_result,))),
                    self.accuracy.get_output_print_call(self.function_name, expected_values),
                    Return(Constant(1, precision = ML_Int32))
                )
            check_array_loop = Loop(
                ReferenceAssign(vj, 0),
                vj < array_len,
                Statement(
                    ConditionBlock(
                        self.accuracy.get_output_check_test(
                            local_result,
                            expected_values
                        ),
                        return_statement_break
                    ),
                    ReferenceAssign(vj, vj+1),
                )
            )
            check_statement.add(check_array_loop)
        return check_statement

    def tensor_element_emulate(self, tensor_descriptors, output_tensor_id, element_id, input_tables):
        raise NotImplementedError


    def generate_expected_table(self, tensor_descriptors, input_tables):
        """ Generate the complete table of expected results """
        # tensor_descriptors unpacking
        (input_tensor_descriptor_list, output_tensor_descriptor_list) = tensor_descriptors
        ## output values required to check results are stored in output table
        NUM_INPUT_ARRAY = len(input_tables)



        expected_tables = []
        # generating expected value table
        for output_id, output_td in enumerate(output_tensor_descriptor_list):
            table_size = output_td.get_bounding_size()

            def expected_value_gen(element_id):
                """ generate a full row of expected values using inputs
                    from input_tables """
                exact_value = self.tensor_element_emulate(tensor_descriptors, output_id, element_id, input_tables)
                output_values = self.accuracy.get_output_check_value(exact_value)
                return output_values

            # FIXME: num_output_value should depend on accuracy requirement
            # for a specific output tensor
            num_output_value = self.accuracy.get_num_output_value()
            expected_table = generate_2d_table(
                table_size, num_output_value,
                output_td.scalar_format,
                "expected_table",
                value_gen=expected_value_gen
            )
            expected_tables.append(expected_table)
        return expected_tables
