# -*- coding: utf-8 -*-
# This file is part of metalibm (https://github.com/kalray/metalibm)

# MIT License
#
# Copyright (c) 2018 Kalray
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import argparse
import sys
import re

from sollya import Interval, SollyaObject


# import meta-tensor kernel scripts from current directory
import mm_kernel

from metalibm_core.core.ml_formats import ML_Int32, ML_Binary32
from metalibm_core.core.precisions import ML_CorrectlyRounded



from metalibm_core.utility.ml_template import (
    target_instanciate, VerboseAction, ExitOnErrorAction)

from valid.test_utils import *


# target instanciation

# list of non-regression tests
# details on NewSchemeTest object can be found in valid.test_utils module
#   Each object requires a title, a function constructor and a list
#   of test cases (each is a dictionnary of parameters -> values)
tensor_test_list = [
  NewSchemeTest(
    "basic matrix multiply kernel",
    mm_kernel.MatrixMultiplyKernel,
    [
        {
        "precision": ML_Int32, "auto_test": 1, "execute_trigger": True,
        "test_index_range": [[14, 14], [10,10], [5, 5]],
        "auto_test_range": [Interval(-10,10), Interval(-10,10)],
        "accuracy": ML_CorrectlyRounded
        },
        {
        "precision": ML_Binary32, "auto_test": 1, "execute_trigger": True,
        "output_file": "mm_kernel_fp32.c",
        "test_index_range": [[14, 14], [10,10], [5, 5]],
        "auto_test_range": [Interval(-10,10), Interval(-10,10)],
        "accuracy": ML_CorrectlyRounded
        },
     ]
  ),
  NewSchemeTest(
    "matrix multiply kernel with numerical simplification",
    mm_kernel.MatrixMultiplyKernel,
    [{
        "precision": ML_Int32, "auto_test": 128, "execute_trigger": True,
        "test_index_range": [[14, 14], [10,10], [5, 5]],
        "auto_test_range": [Interval(-10,10), Interval(-10,10)],
        "extra_passes": [("beforecodegen:numerical_simplification")],
        "accuracy": ML_CorrectlyRounded
        },
     ]
  ),
  NewSchemeTest(
    "matrix multiply kernel with vectorization",
    mm_kernel.MatrixMultiplyKernel,
    [{
        "precision": ML_Binary32, "auto_test": 128, "execute_trigger": True,
        "test_index_range": [[16, 16], [16,16], [16, 16]],
        "auto_test_range": [Interval(-10,10), Interval(-10,10)],
        "output_file": "mm_kernel_vectorized.c",
        "accuracy": ML_CorrectlyRounded,
        "vectorize": True,
        "target": target_instanciate("vector"),
        },
     ]
  ),
]

test_tag_map = {}
for test in tensor_test_list:
  test_tag_map[test.get_tag_title()] = test

## Command line action to set break on error in load module
class ListTestAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        for test in  tensor_test_list:
          print(test.get_tag_title())
        exit(0)

# generate list of test object from string
# of comma separated test's tag
def parse_test_list(test_list):
  test_tags = test_list.split(",")
  return [test_tag_map[tag] for tag in test_tags]

arg_parser = argparse.ArgumentParser(" Metalibm's tensor kernel codegen non-regression tests")
# enable debug mode
arg_parser.add_argument("--debug", dest = "debug", action = "store_const",
                        default = False, const = True,
                        help = "enable debug mode")
# listing available tests
arg_parser.add_argument("--list", action = ListTestAction, help = "list available test", nargs = 0)

# select list of tests to be executed
arg_parser.add_argument("--execute", dest = "test_list", type = parse_test_list, default = tensor_test_list, help = "list of comma separated test to be executed")

arg_parser.add_argument("--match", dest = "match_regex", type = str, default = ".*", help = "list of comma separated match regexp to be used for test selection")
arg_parser.add_argument(
    "--exit-on-error", dest="exit_on_error",
    action=ExitOnErrorAction, const=True,
    default=False,
    nargs=0,
    help="convert Fatal error to sys exit rather than exception")


arg_parser.add_argument(
    "--verbose", dest="verbose_enable", action=VerboseAction,
    const=True, default=False,
    help="enable Verbose log level")


args = arg_parser.parse_args(sys.argv[1:])

success = True
# list of TestResult objects generated by execution
# of new scheme tests
result_details = []

for test_scheme in args.test_list:
  if re.search(args.match_regex, test_scheme.get_tag_title()) != None:
    test_result = test_scheme.perform_all_test(debug = args.debug)
    result_details.append(test_result)
    if not test_result.get_result():
      success = False

unexpected_failure_count = 0

# Printing test summary for new scheme
for result in result_details:
  print(result.get_details())
  unexpected_failure_count += result.unexpected_count

print(" {} unexpected failure(s)".format(unexpected_failure_count))

if success:
  print("OVERALL SUCCESS")
  exit(0)
else:
  print("OVERALL FAILURE")
  exit(1)
