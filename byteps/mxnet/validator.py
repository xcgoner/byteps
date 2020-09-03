# Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import mxnet as mx

class Validator:
    def validate(self, update, result, validation_info = None):
        """

        Parameters
        ----------
        update
            single tensor or a list of tensors
        result
            single tensor to store the result
        validation_info
            additional parameters for validation, e.g., validation tensors
        """
        raise NotImplementedError

class NaiveValidator(Validator):
    # implement simple average
    def validate(self, update, result, validation_info = None):
        # update sould be a list of tensors
        # validation_info contains the number of workers
        assert isinstance(
        update, list
        ), "Input must be a list of tensors"
        assert len(update) > 0
        result[:] = update[0]
        for i in range(1, validation_info['num_tensors']):
            result[:] += update[i]
        result /= validation_info['num_tensors']

class TrimmedMeanValidator(Validator):
    def __init__(self, num_trimmed = 1):
        self.num_trimmed = num_trimmed
    # implement simple average
    def validate(self, update, result, validation_info = None):
        # update sould be a list of tensors
        # validation_info contains the number of workers
        assert isinstance(
        update, list
        ), "Input must be a list of tensors"
        assert validation_info['num_tensors'] > 2 * self.num_trimmed
        result[:] = mx.nd.stack(*update).sort(axis=0)[self.num_trimmed:(validation_info['num_tensors']-self.num_trimmed),:].mean(axis=0)

class PhocasValidator(Validator):
    def __init__(self, num_trimmed = 1):
        self.num_trimmed = num_trimmed
    # implement simple average
    def validate(self, update, result, validation_info = None):
        # update sould be a list of tensors
        # validation_info contains the number of workers
        assert isinstance(
        update, list
        ), "Input must be a list of tensors"
        assert validation_info['num_tensors'] > 2 * self.num_trimmed
        sorted_array = mx.nd.stack(*update[:validation_info['num_tensors']]).sort(axis=0)
        trimmed_mean = sorted_array[self.num_trimmed:(validation_info['num_tensors']-self.num_trimmed),:].mean(axis=0, keepdims=1)
        result[:] = mx.nd.sum(sorted_array * mx.nd.topk(mx.nd.abs(sorted_array-trimmed_mean), ret_typ='mask', k=(validation_info['num_tensors']-self.num_trimmed), is_ascend=1, axis=0), axis=0) / float(validation_info['num_tensors']-self.num_trimmed)

class ZenoValidator(Validator):
    def __init__(self, eta = 0.1, rho = 0.1):
        self.eta = eta
        self.rho = rho
    # implement simple average
    def validate(self, update, result, validation_info = None):
        # update sould be a list of tensors
        # validation_info contains the validation tensor
        assert isinstance(
        update, list
        ), "Input must be a list of tensors"
        counter = 0
        for u in update[:validation_info['num_tensors']]:
            a = (u * validation_info['validation_tensor']).mean().asscalar()
            b = validation_info['validation_tensor'].square().mean().asscalar()
            c = u.square().mean().asscalar()
            if np.isnan(a):
                print(u, validation_info['validation_tensor'])
                assert not np.isnan(b)
            if a > b * self.eta and c < b * (1 + self.rho):
                if counter == 0:
                    result[:] = u
                else:
                    result[:] += u
                counter += 1
        if counter > 0:
            result /= validation_info['num_tensors']
        else:
            result[:] = 0
        return counter

class ZenoppValidator(Validator):
    def __init__(self, eta = 0.1, rho = 0.1, alpha = 0.8):
        self.eta = eta
        self.rho = rho
        self.alpha = alpha
    # implement simple average
    def validate(self, update, result, validation_info = None):
        # update sould be single tensor
        # validation_info contains the validation tensor
        a = (update * validation_info['validation_tensor']).mean().asscalar()
        b = validation_info['validation_tensor'].square().mean().asscalar()
        c = update.square().mean().asscalar()
        if a > b * self.eta and c < b * (1 + self.rho):
            result[:] = update * (self.alpha / (1 + c))
            return 1
        else:
            result[:] = 0
            return 0

class NaiveAsyncValidator(Validator):
    def __init__(self, alpha = 0.8):
        self.alpha = alpha
    # implement simple average
    def validate(self, update, result, validation_info = None):
        # update sould be single tensor
        # validation_info contains the validation tensor
        result[:] = update * self.alpha
        


