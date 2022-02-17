// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/mlir/disc/tools/disc-replay/disc_interpreter.h"

#include <dlfcn.h>

#include <iostream>

namespace replay {

#if GOOGLE_CUDA
using ::stream_executor::gpu::GpuDevicePtr;
using ::stream_executor::gpu::GpuStatus;

#define GPU_SUCCESS CUDA_SUCCESS
#define GPU_MEMCPYDTOH_API cuMemcpyDtoH

static void printErrorIfAny(GpuStatus result, const char* where) {
  if (result != GPU_SUCCESS) {
    std::ostringstream out;
    LOG(ERROR) << "CUDA failed with " << result << " in " << where;
  }
}

static int32_t reportErrorIfAny(GpuStatus result, const char* where) {
  printErrorIfAny(result, where);
  return result;
}
#endif

DiscInterpreter::DiscInterpreter() {
  ral_func_ptr_ = reinterpret_cast<void*>(&tao_ral_call_impl);
}

tensorflow::Status DiscInterpreter::Compile(
    tensorflow::tao::TaoCompilerInput& input, CompiledResult& result) {
  auto env = tensorflow::Env::Default();
  std::string tmp_file;
  env->LocalTempFilename(&tmp_file);

  // compile input proto to executable file
  tensorflow::DeviceType device_type(input.options().device_type());
  auto* compiler_wrapper =
      tensorflow::tao::CompilerBase::GetCompilerForDevice(device_type)
          .ConsumeValueOrDie();
  TF_RETURN_IF_ERROR(compiler_wrapper->Compile(input, tmp_file));
  result.output_fname = tmp_file + ".so";
  result.meta_fname = tmp_file + ".so.pbtxt";
  TF_RETURN_IF_ERROR(GetEntryFunc(result.output_fname, result.entry_func));
  InitExecCUDAContext(result.meta_fname);

  return tensorflow::Status::OK();
}

tensorflow::Status BindInputs(const std::vector<tensorflow::Tensor>& tensors,
                              const std::vector<std::string> placements,
                              tao::ral::ExecutionContext& exec_ctx) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto t = tensors[i];
    // for debug
    if (i == 120) {
      std::cout << "%arg120 shape: [";
      for (int r = 0; r < t.dims(); ++r) {
        std::cout << t.dim_size(r) << ", ";
      }
      std::cout << "]: ";
      for (int n = 0; n < t.NumElements(); ++n) {
        std::cout << reinterpret_cast<float*>(t.data())[n] << ", ";
      }
      std::cout << std::endl;
    }
    std::vector<int64_t> shape;
    for (size_t dim_i = 0; dim_i < t.dims(); ++dim_i) {
      shape.push_back(t.dim_size(dim_i));
    }
#if GOOGLE_CUDA
    if (placements[i] == "cpu") {
      exec_ctx.bindInput(i, t.data(), shape);
    } else {
      void* d_addr = nullptr;
      auto result = cuMemAlloc((GpuDevicePtr*)&d_addr, t.TotalBytes());
      if (result != CUDA_SUCCESS) {
        return tensorflow::errors::Internal("cuda memory alloc failed");
      }
      result = cuMemcpyHtoD((GpuDevicePtr)d_addr, t.data(), t.TotalBytes());
      if (result != CUDA_SUCCESS) {
        return tensorflow::errors::Internal("cuda memcpy H2D failed");
      }
      exec_ctx.bindInput(i, d_addr, shape);
    }
#else
    if (placements[i] != "cpu") {
      return tensorflow::errors::Internal(
          "unexpected input placement, only host tag for CPU only build");
    } else {
      exec_ctx.bindInput(i, t.data(), shape);
    }
#endif
  }
  return tensorflow::Status::OK();
}

tensorflow::Status BindOutputsAndDump(const std::vector<std::string> placements,
                                      tao::ral::ExecutionContext& exec_ctx) {
  size_t num_outputs = placements.size();
  for (size_t idx = 0; idx < num_outputs; ++idx) {
    std::unique_ptr<tao::ral::OutputBufferWrapper> output_buffer;
    output_buffer.reset();
    exec_ctx.bindOutput(idx, &output_buffer);
    void* result = (void*)output_buffer->data();
    auto& output_shape = output_buffer->shape();
    int64_t nelem = 1;
    for (size_t i = 0; i < output_shape.size(); ++i) {
      nelem *= output_shape[i];
    }
    // if (out_elem_types_[idx] == tensorflow::DT_FLOAT) {
    // TODO: only support float output for now
    if (true) {
      if (placements[idx] == "cpu") {
        std::cout << "  result #" << idx << ": shape: [";
        for (int i = 0; i < output_shape.size(); ++i) {
          std::cout << output_shape[i] << ", ";
        }
        std::cout << "] ";
        for (int i = 0; i < nelem && i < 5; ++i) {
          std::cout << reinterpret_cast<float*>(result)[i] << ", ";
        }
        std::cout << std::endl;
      } else if (placements[idx] == "gpu") {
        int64_t bytes = nelem * sizeof(float);
        float* h_result = nullptr;
        if (nelem) {
          h_result = new float[nelem];
          reportErrorIfAny(
              GPU_MEMCPYDTOH_API((void*)h_result,
                                 reinterpret_cast<GpuDevicePtr>(result), bytes),
              "gpu MemcpyDtoH");
        }
        std::cout << "  result #" << idx << ": shape: [";
        for (int i = 0; i < output_shape.size(); ++i) {
          std::cout << output_shape[i] << ", ";
        }
        std::cout << "] ";
        for (int i = 0; i < nelem && (i < 5 || true); ++i) {
          std::cout << h_result[i] << ", ";
        }
        std::cout << std::endl;
      } else {
        std::cout << "unexpected output placement: " << placements[idx]
                  << std::endl;
      }
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status DiscInterpreter::Run(
    const CompiledResult& result,
    const std::vector<tensorflow::Tensor>& tensors,
    const std::vector<std::string>& input_placements,
    const std::vector<std::string>& output_placements) {
#if GOOGLE_CUDA
  auto exec_ctx =
      tao::ral::MakeExecutionContext<tao::ral::gpu::BaseCudaExecutionContext>(
          context_.get());
#else
  auto exec_ctx =
      tao::ral::MakeExecutionContext<tao::ral::cpu::BaseCpuExecutionContext>(
          context_.get());
#endif
  TF_RETURN_IF_ERROR(BindInputs(tensors, input_placements, *exec_ctx.get()));
  void* ctx_struct[] = {exec_ctx.get(), ral_func_ptr_};
  result.entry_func(ctx_struct);
  // TODO: support correctness check
  TF_RETURN_IF_ERROR(BindOutputsAndDump(output_placements, *exec_ctx.get()));
  return tensorflow::Status::OK();
}

void DiscInterpreter::InitExecCUDAContext(const std::string& meta_fname) {
  tao::ral::BaseContextOption opt;
  opt.metadata_file_path = meta_fname;
  opt.cache_workspace_mem_across_execution = true;
  tao::ral::cpu::BaseCpuContextOption cpu_opt;
#if GOOGLE_CUDA
  tao::ral::gpu::BaseCudaContextOption gpu_opt;
  gpu_opt.use_stream_executor = true;
  context_ = tao::ral::gpu::MakeBaseCudaContext(opt, cpu_opt, gpu_opt);
#else
  context_ = tao::ral::cpu::MakeBaseCpuContext(opt, cpu_opt);
#endif
}

tensorflow::Status DiscInterpreter::GetEntryFunc(
    const std::string& exectuable_fname, func_t& entry_func) {
  void* func_handle = dlopen(exectuable_fname.c_str(), RTLD_NOW);
  if (!func_handle) {
    std::string msg = "fail to open compiled .so file with error: ";
    absl::StrAppend(&msg, dlerror());
    return tensorflow::errors::Internal(msg);
  }

  void* entry_func_ptr = dlsym(func_handle, "main");
  if (!entry_func_ptr) {
    return tensorflow::errors::Internal("fail to find the main");
  }
  entry_func = (func_t)entry_func_ptr;
  CHECK_NOTNULL(entry_func);
  return tensorflow::Status::OK();
}

}  //  namespace replay
