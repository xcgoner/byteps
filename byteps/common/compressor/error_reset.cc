// Copyright 2019 Amazon Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "error_reset.h"

namespace byteps {
namespace common {
namespace compressor {

tensor_t ErrorReset::Compress(tensor_t grad) {
  // 1. grad <- grad + error
  UpdateGradient(grad);

  // 2. c <- Compress(grad)
  auto compressed = _cptr->Compress(grad);

  // 3. e <- grad - Decompress(c)
  UpdateError(grad, compressed);

  return compressed;
}

tensor_t ErrorReset::Decompress(tensor_t compressed) {
  // directly forward to internal compressor
  auto decompressed = _cptr->Decompress(compressed);
  // decompressed
  this->_cpu_reducer->sum(decompressed.data, _prev_error.get(), decompressed.size,
                          static_cast<DataType>(decompressed.dtype),
                          -1);
  this->_cpu_reducer->sum(decompressed.data, _error.get(), decompressed.size,
                          static_cast<DataType>(decompressed.dtype));
  return decompressed
}

void ErrorReset::UpdateError(tensor_t corrected, tensor_t compressed) {
  // cache the previous error before update the new error
  std::memcpy(_prev_error.get(), _error.get(), _size);
  tensor_t error{_error.get(), _size, corrected.dtype};
  _cptr->FastUpdateError(error, corrected, compressed);
}

}  // namespace compressor
}  // namespace common
}  // namespace byteps