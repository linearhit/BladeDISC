package(default_visibility = ["//visibility:public"])
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda_is_configured",
)
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_import")
load("//bazel:build_defs.bzl", "if_tensorrt_enabled", "if_platform_alibaba")

exports_files([
    "pybind.h",
])

cc_library(
    name = "torch_blade_include",
    includes = ["."],
    visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "torch_blade",
    deps = [
        "//src/common_utils:torch_blade_common",
	"//src/compiler/backends:torch_blade_backends",
        "//src/compiler/jit:torch_blade_jit",
        "//src/compiler/mlir:torch_blade_mlir",
        "@local_org_torch//:libtorch",
    ] + if_tensorrt_enabled(["//src/compiler/tensorrt:tensorrt_runtime_impl"])
      + if_cuda_is_configured([
        "@local_config_cuda//cuda:cudart",
    ]) + if_platform_alibaba(["//src/platform_alibaba:torch_blade",]),
    linkopts = ["-Wl,-rpath,$$ORIGIN"],
)

cc_binary(
    name = "libtorch_blade.so",
    deps = [":torch_blade"],
    linkshared = 1,
)

cc_import(
    name = "torch_blade_shared",
    shared_library = "libtorch_blade.so",
)

# Pybind11 bindings for TorchBlade

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

licenses(["notice"])

exports_files(["LICENSE"])

filegroup(
    name = "torch_blade_py_sources",
    srcs = glob(["*pybind*.cpp"]) + ["//src/compiler:torch_blade_py_srcs"],
)

filegroup(
    name = "torch_blade_py_headers",
    srcs = glob(["*.h"]) + ["//src/compiler:torch_blade_py_hdrs"],
)

pybind_library(
    name = "torch_blade_pybind11",
    srcs = [
        ":torch_blade_py_sources",
    ],
    hdrs = [
        ":torch_blade_py_headers",
    ],
    includes = ["."],
    linkstatic = 0,
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":torch_blade_shared",
        "@local_org_torch//:torch_python",
    ] + if_tensorrt_enabled(["@tensorrt//:nvinfer_headers"]),
    copts = ["-DTORCH_BLADE_BUILD_MLIR"] +
      if_tensorrt_enabled(["-DTORCH_BLADE_BUILD_TENSORRT"]) +
      if_platform_alibaba(
        ["-DTORCH_BLADE_PLATFORM_ALIBABA=true"],
        ["-DTORCH_BLADE_PLATFORM_ALIBABA=false"]
      ),
)

pybind_extension(
    name = "_torch_blade",
    linkopts = ["-Wl,-rpath,$$ORIGIN"],
    deps = [
        ":torch_blade_shared",
        ":torch_blade_pybind11",
    ] + if_platform_alibaba(["//src/platform_alibaba:torch_blade_pybind11",]),
)
