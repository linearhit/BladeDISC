// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <mlir/mhlo/builder/softmax.h>

#include "common_utils/logging.h"
#include "compiler/mlir/converters/impl/prim_constant.h"
#include "compiler/mlir/converters/impl/utils.h"
#include "compiler/mlir/converters/mhlo_converter_register.h"
#include "compiler/mlir/converters/mlir_type_utils.h"
#include "mlir/mhlo/builder/mlir_shape_builder.h"
#include "mlir/mhlo/builder/mlir_utils.h"

#include <torch/script.h>

namespace torch {
namespace blade {
using namespace mlir::mhlo;

template <bool is_logsoftmax = false>
bool ConvertAtenSoftmax(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  const auto& ml_raw_input = ctx.GetMlirValue(node.input(0));
  auto jit_dim = node.input(1);
  auto jit_dtype = node.input(2);
  static constexpr const char* op_name = "aten::softmax";
  if (!CheckConstAttribute(jit_dim, op_name, "dim")) {
    return false;
  }
  if (!CheckConstAttribute(jit_dtype, op_name, "dtype")) {
    return false;
  }

  auto jit_dim_ival = torch::jit::toIValue(jit_dim);
  mlir_dim_t reduce_dim = -1;
  if (jit_dim_ival && !jit_dim_ival->isNone()) {
    reduce_dim = CastJitConstToInt64(*jit_dim);
  }
  auto& builder = *ctx.builder;
  auto optional_input_casted =
      BuildCastWithJitType(builder, loc, ml_raw_input, jit_dtype);
  if (!optional_input_casted) {
    TORCH_CHECK(jit_dtype != nullptr);
    DLOG(INFO)
        << "Could not convert aten::softmax with invalid parameter: dtype %"
        << jit_dtype->debugName();
    return false;
  }

  auto ml_input = *optional_input_casted;

  // reformat the reduce dim into the last dim for performance
  mlir_dim_t rank = GetRankOfMlirValue(ml_input);
  reduce_dim = NormalizeDimIndex(reduce_dim, rank);
  mlir::Value softmax_input = ml_input;
  bool need_transpose = (reduce_dim != rank - 1);
  mlir::Value result;
  if (need_transpose) {
    SmallVec4<mlir_dim_t> permutation;
    SmallVec4<mlir_dim_t> rev_permutation(rank, -1);
    for (mlir_dim_t i = 0; i < rank; ++i) {
      if (i != reduce_dim) {
        permutation.push_back(i);
        rev_permutation[i] = permutation.size() - 1;
      }
    }
    permutation.push_back(reduce_dim);
    rev_permutation[reduce_dim] = rank - 1;
    softmax_input = BuildPermute(builder, loc, softmax_input, permutation);
    result = BuildSoftmax(builder, loc, softmax_input, rank - 1, is_logsoftmax);
    result = BuildPermute(builder, loc, result, rev_permutation);
  } else {
    result =
        BuildSoftmax(builder, loc, softmax_input, reduce_dim, is_logsoftmax);
  }
  ctx.value_map[node.output(0)] = result;
  return true;
}

namespace {
auto mhlo_conversion =
    MhloConversionPatternRegister()
        .pattern(
            "aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
            ConvertAtenSoftmax)
        .pattern(
            "aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
            ConvertAtenSoftmax<true>);
} // namespace
} // namespace blade
} // namespace torch
