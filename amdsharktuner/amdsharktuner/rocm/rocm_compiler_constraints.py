# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Integration module for using compiler-side SMT constraints (from
LLVMGPUConstraintGenerator) to enumerate valid GPU pipeline configurations.

Uses the compiler's constraint generation + SMT-LIB export + Z3 solving.
Supports VectorDistribute for matmul, conv, and attention ops,
plus TileAndFuse for matmul and conv.
"""

import math
import re
import shutil
import subprocess
from typing import Callable, Iterator, Optional

import z3  # type: ignore

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, iree_gpu  # type: ignore

from .. import common, dispatch_parser
from . import rocm_common


def build_hal_executable_mlir_str(
    m: int,
    n: int,
    k: int,
    lhs_elem: str,
    rhs_elem: str,
    res_elem: str,
    gpu_target_info: iree_gpu.TargetInfo,
) -> str:
    """
    Build MLIR text for a hal.executable module containing a matmul,
    suitable for running insert-smt-constraints.

    Args:
        m, n, k: Problem dimensions.
        lhs_elem, rhs_elem, res_elem: Element type strings (e.g. "f16", "f32").
        gpu_target_info: GPU target info for the #iree_gpu.target attribute.

    Returns:
        MLIR module string.
    """
    # Build MMA list string.
    mma_strs = []
    for intrinsic in gpu_target_info.mma_intrinsics:
        mma_strs.append(f"<{intrinsic}>")
    mma_list = ", ".join(mma_strs)

    sg_choices = list(gpu_target_info.subgroup_size_choices)
    sg_choices_str = "[" + ", ".join(str(s) for s in sg_choices) + "]"

    max_wg_sizes = list(gpu_target_info.max_workgroup_sizes)
    max_wg_sizes_str = "[" + ", ".join(str(s) for s in max_wg_sizes) + "]"

    max_threads = gpu_target_info.max_thread_count_per_workgroup
    max_shmem = gpu_target_info.max_workgroup_memory_bytes
    arch = gpu_target_info.arch

    return f"""
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#gpu_target = #iree_gpu.target<arch = "{arch}", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [{mma_list}],
  subgroup_size_choices = {sg_choices_str},
  max_load_instruction_bits = 128,
  max_workgroup_sizes = {max_wg_sizes_str}, max_thread_count_per_workgroup = {max_threads},
  max_workgroup_memory_bytes = {max_shmem},
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {{iree_codegen.target_info = #gpu_target}}>

hal.executable @matmul_ex {{
  hal.executable.variant public @rocm target(#exec_target) {{
    hal.executable.export public @matmul ordinal(0) layout(#pipeline_layout)
    builtin.module {{
      func.func @matmul() {{
        %cst = arith.constant 0.0 : {res_elem}
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<{m}x{k}x{lhs_elem}>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<{k}x{n}x{rhs_elem}>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<{m}x{n}x{res_elem}>>
        %lhs = iree_tensor_ext.dispatch.tensor.load %0, offsets=[0, 0], sizes=[{m}, {k}], strides=[1, 1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<{m}x{k}x{lhs_elem}>> -> tensor<{m}x{k}x{lhs_elem}>
        %rhs = iree_tensor_ext.dispatch.tensor.load %1, offsets=[0, 0], sizes=[{k}, {n}], strides=[1, 1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<{k}x{n}x{rhs_elem}>> -> tensor<{k}x{n}x{rhs_elem}>
        %empty = tensor.empty() : tensor<{m}x{n}x{res_elem}>
        %fill = linalg.fill ins(%cst : {res_elem}) outs(%empty : tensor<{m}x{n}x{res_elem}>) -> tensor<{m}x{n}x{res_elem}>
        %result = linalg.matmul ins(%lhs, %rhs : tensor<{m}x{k}x{lhs_elem}>, tensor<{k}x{n}x{rhs_elem}>)
            outs(%fill : tensor<{m}x{n}x{res_elem}>) -> tensor<{m}x{n}x{res_elem}>
        iree_tensor_ext.dispatch.tensor.store %result, %2, offsets=[0, 0], sizes=[{m}, {n}], strides=[1, 1]
            : tensor<{m}x{n}x{res_elem}> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<{m}x{n}x{res_elem}>>
        return
      }}
    }}
  }}
}}
"""


def build_hal_executable_conv_mlir_str(
    batch: int,
    oh: int,
    ow: int,
    oc: int,
    ih: int,
    iw: int,
    ic: int,
    fh: int,
    fw: int,
    input_elem: str,
    filter_elem: str,
    res_elem: str,
    gpu_target_info: iree_gpu.TargetInfo,
) -> str:
    """
    Build MLIR text for a hal.executable module containing a conv_2d_nhwc_hwcf,
    suitable for running insert-smt-constraints.

    Args:
        batch, oh, ow, oc: Output dimensions (batch, output height/width, output channels).
        ih, iw, ic: Input dimensions (input height/width, input channels).
        fh, fw: Filter dimensions (filter height/width).
        input_elem, filter_elem, res_elem: Element type strings (e.g. "f16", "f32").
        gpu_target_info: GPU target info for the #iree_gpu.target attribute.

    Returns:
        MLIR module string.
    """
    mma_strs = []
    for intrinsic in gpu_target_info.mma_intrinsics:
        mma_strs.append(f"<{intrinsic}>")
    mma_list = ", ".join(mma_strs)

    sg_choices = list(gpu_target_info.subgroup_size_choices)
    sg_choices_str = "[" + ", ".join(str(s) for s in sg_choices) + "]"

    max_wg_sizes = list(gpu_target_info.max_workgroup_sizes)
    max_wg_sizes_str = "[" + ", ".join(str(s) for s in max_wg_sizes) + "]"

    max_threads = gpu_target_info.max_thread_count_per_workgroup
    max_shmem = gpu_target_info.max_workgroup_memory_bytes
    arch = gpu_target_info.arch

    return f"""
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#gpu_target = #iree_gpu.target<arch = "{arch}", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [{mma_list}],
  subgroup_size_choices = {sg_choices_str},
  max_load_instruction_bits = 128,
  max_workgroup_sizes = {max_wg_sizes_str}, max_thread_count_per_workgroup = {max_threads},
  max_workgroup_memory_bytes = {max_shmem},
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {{iree_codegen.target_info = #gpu_target}}>

hal.executable @conv_ex {{
  hal.executable.variant public @rocm target(#exec_target) {{
    hal.executable.export public @conv2d ordinal(0) layout(#pipeline_layout)
    builtin.module {{
      func.func @conv2d() {{
        %cst = arith.constant 0.0 : {res_elem}
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<{batch}x{ih}x{iw}x{ic}x{input_elem}>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<{fh}x{fw}x{ic}x{oc}x{filter_elem}>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<{batch}x{oh}x{ow}x{oc}x{res_elem}>>
        %input = iree_tensor_ext.dispatch.tensor.load %0, offsets=[0,0,0,0], sizes=[{batch},{ih},{iw},{ic}], strides=[1,1,1,1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<{batch}x{ih}x{iw}x{ic}x{input_elem}>> -> tensor<{batch}x{ih}x{iw}x{ic}x{input_elem}>
        %filter = iree_tensor_ext.dispatch.tensor.load %1, offsets=[0,0,0,0], sizes=[{fh},{fw},{ic},{oc}], strides=[1,1,1,1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<{fh}x{fw}x{ic}x{oc}x{filter_elem}>> -> tensor<{fh}x{fw}x{ic}x{oc}x{filter_elem}>
        %empty = tensor.empty() : tensor<{batch}x{oh}x{ow}x{oc}x{res_elem}>
        %fill = linalg.fill ins(%cst : {res_elem}) outs(%empty : tensor<{batch}x{oh}x{ow}x{oc}x{res_elem}>) -> tensor<{batch}x{oh}x{ow}x{oc}x{res_elem}>
        %result = linalg.conv_2d_nhwc_hwcf {{
            dilations = dense<1> : tensor<2xi64>,
            strides = dense<1> : tensor<2xi64>
        }} ins(%input, %filter : tensor<{batch}x{ih}x{iw}x{ic}x{input_elem}>, tensor<{fh}x{fw}x{ic}x{oc}x{filter_elem}>)
            outs(%fill : tensor<{batch}x{oh}x{ow}x{oc}x{res_elem}>) -> tensor<{batch}x{oh}x{ow}x{oc}x{res_elem}>
        iree_tensor_ext.dispatch.tensor.store %result, %2, offsets=[0,0,0,0], sizes=[{batch},{oh},{ow},{oc}], strides=[1,1,1,1]
            : tensor<{batch}x{oh}x{ow}x{oc}x{res_elem}> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<{batch}x{oh}x{ow}x{oc}x{res_elem}>>
        return
      }}
    }}
  }}
}}
"""


def build_hal_executable_attention_mlir_str(
    batch: int,
    seq_len: int,
    head_dim_k1: int,
    kv_seq_len: int,
    head_dim_n: int,
    q_elem: str,
    k_elem: str,
    v_elem: str,
    res_elem: str,
    gpu_target_info: iree_gpu.TargetInfo,
) -> str:
    """
    Build MLIR text for a hal.executable module containing an attention op,
    suitable for running insert-smt-constraints.

    The attention iteration domain is [B, M, K1, K2, N] with:
        Q: B x M x K1
        K: B x K2 x K1
        V: B x K2 x N
        O: B x M x N

    Args:
        batch: Batch dimension.
        seq_len: Query sequence length (M).
        head_dim_k1: Head dimension for QK matmul (K1).
        kv_seq_len: Key/value sequence length (K2).
        head_dim_n: Head dimension for PV matmul / output (N).
        q_elem, k_elem, v_elem, res_elem: Element type strings.
        gpu_target_info: GPU target info for the #iree_gpu.target attribute.

    Returns:
        MLIR module string.
    """
    mma_strs = []
    for intrinsic in gpu_target_info.mma_intrinsics:
        mma_strs.append(f"<{intrinsic}>")
    mma_list = ", ".join(mma_strs)

    sg_choices = list(gpu_target_info.subgroup_size_choices)
    sg_choices_str = "[" + ", ".join(str(s) for s in sg_choices) + "]"

    max_wg_sizes = list(gpu_target_info.max_workgroup_sizes)
    max_wg_sizes_str = "[" + ", ".join(str(s) for s in max_wg_sizes) + "]"

    max_threads = gpu_target_info.max_thread_count_per_workgroup
    max_shmem = gpu_target_info.max_workgroup_memory_bytes
    arch = gpu_target_info.arch

    # Scale is typically 1/sqrt(head_dim_k1).
    # We just need a valid f16 constant for the MLIR to parse.
    return f"""
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#gpu_target = #iree_gpu.target<arch = "{arch}", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [{mma_list}],
  subgroup_size_choices = {sg_choices_str},
  max_load_instruction_bits = 128,
  max_workgroup_sizes = {max_wg_sizes_str}, max_thread_count_per_workgroup = {max_threads},
  max_workgroup_memory_bytes = {max_shmem},
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {{iree_codegen.target_info = #gpu_target}}>

hal.executable @attention_ex {{
  hal.executable.variant public @rocm target(#exec_target) {{
    hal.executable.export public @attention ordinal(0) layout(#pipeline_layout)
    builtin.module {{
      func.func @attention() {{
        %cst = arith.constant 1.250000e-01 : {q_elem}
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<{batch}x{seq_len}x{head_dim_k1}x{q_elem}>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<{batch}x{kv_seq_len}x{head_dim_k1}x{k_elem}>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<{batch}x{kv_seq_len}x{head_dim_n}x{v_elem}>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<{batch}x{seq_len}x{head_dim_n}x{res_elem}>>
        %q = iree_tensor_ext.dispatch.tensor.load %0, offsets=[0,0,0], sizes=[{batch},{seq_len},{head_dim_k1}], strides=[1,1,1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<{batch}x{seq_len}x{head_dim_k1}x{q_elem}>> -> tensor<{batch}x{seq_len}x{head_dim_k1}x{q_elem}>
        %k = iree_tensor_ext.dispatch.tensor.load %1, offsets=[0,0,0], sizes=[{batch},{kv_seq_len},{head_dim_k1}], strides=[1,1,1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<{batch}x{kv_seq_len}x{head_dim_k1}x{k_elem}>> -> tensor<{batch}x{kv_seq_len}x{head_dim_k1}x{k_elem}>
        %v = iree_tensor_ext.dispatch.tensor.load %2, offsets=[0,0,0], sizes=[{batch},{kv_seq_len},{head_dim_n}], strides=[1,1,1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<{batch}x{kv_seq_len}x{head_dim_n}x{v_elem}>> -> tensor<{batch}x{kv_seq_len}x{head_dim_n}x{v_elem}>
        %empty = tensor.empty() : tensor<{batch}x{seq_len}x{head_dim_n}x{res_elem}>
        %result = iree_linalg_ext.attention {{
            indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                             affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                             affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                             affine_map<(d0, d1, d2, d3, d4) -> ()>,
                             affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}}
            ins(%q, %k, %v, %cst : tensor<{batch}x{seq_len}x{head_dim_k1}x{q_elem}>, tensor<{batch}x{kv_seq_len}x{head_dim_k1}x{k_elem}>, tensor<{batch}x{kv_seq_len}x{head_dim_n}x{v_elem}>, {q_elem})
            outs(%empty : tensor<{batch}x{seq_len}x{head_dim_n}x{res_elem}>) {{
            ^bb0(%score: f32):
              iree_linalg_ext.yield %score : f32
            }} -> tensor<{batch}x{seq_len}x{head_dim_n}x{res_elem}>
        iree_tensor_ext.dispatch.tensor.store %result, %3, offsets=[0,0,0], sizes=[{batch},{seq_len},{head_dim_n}], strides=[1,1,1]
            : tensor<{batch}x{seq_len}x{head_dim_n}x{res_elem}> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<{batch}x{seq_len}x{head_dim_n}x{res_elem}>>
        return
      }}
    }}
  }}
}}
"""


def run_insert_smt_constraints(
    module_str: str, iree_opt_path: Optional[str] = None
) -> str:
    """
    Run iree-opt with the insert-smt-constraints pass pipeline.

    Args:
        module_str: MLIR text of a hal.executable module.
        iree_opt_path: Path to iree-opt binary. Uses shutil.which if None.

    Returns:
        Output MLIR text with iree_codegen.smt.constraints ops inserted.
    """
    if iree_opt_path is None:
        iree_opt_path = shutil.which("iree-opt")
    if iree_opt_path is None:
        raise RuntimeError(
            "iree-opt not found on PATH. Set iree_opt_path or add build/tools to PATH."
        )

    cmd = [
        iree_opt_path,
        "--iree-codegen-add-tuner-attributes",
        "--pass-pipeline=builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-llvmgpu-select-lowering-strategy,func.func(iree-codegen-insert-smt-constraints)))))",
    ]
    result = subprocess.run(
        cmd,
        input=module_str,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"iree-opt failed (exit code {result.returncode}):\n{result.stderr}"
        )
    return result.stdout


def _make_module_parseable(module_str: str) -> str:
    """
    Preprocess the output from iree-opt to replace unparseable attributes.

    Older DispatchLoweringPassPipelineAttr had an empty mnemonic and could not be
    round-tripped through text. Replace it with a string attr so the
    module can be parsed by the Python API.
    """
    # Replace #iree_codegen< LLVMGPUVectorDistribute> with "LLVMGPUVectorDistribute".
    return re.sub(
        r"#iree_codegen<\s*(\w+)\s*>",
        r'"\1"',
        module_str,
    )


def _get_constraints_ops(input_module: ir.Module):
    return ir.get_ops_of_type(input_module, iree_codegen.ConstraintsOp)


def _constraints_op_to_smtlib(constraints_op) -> str:
    return iree_codegen.convert_constraints_op_to_smtlib(
        constraints_op, emit_reset=False
    )


def extract_smtlib_from_module(
    module_str: str, context: ir.Context
) -> list[tuple[str, ir.DictAttr]]:
    """
    Parse a module containing iree_codegen.smt.constraints ops and extract
    SMT-LIB strings + knobs metadata.

    Args:
        module_str: MLIR text of module with constraints ops.
        context: MLIR context to use for parsing.

    Returns:
        List of (smtlib_str, knobs_dict_attr) tuples.
    """
    parseable_str = _make_module_parseable(module_str)
    with context:
        input_module = ir.Module.parse(parseable_str)
        constraints_ops = _get_constraints_ops(input_module)

        results = []
        for op in constraints_ops:
            smtlib = _constraints_op_to_smtlib(op)
            knobs = ir.DictAttr(op.attributes["knobs"])
            results.append((smtlib, knobs))

        return results


def _collect_knob_names(attr, names: list[str]):
    """Recursively walk an attribute tree collecting knob names."""
    if iree_codegen.IntKnobAttr.isinstance(attr):
        names.append(iree_codegen.IntKnobAttr(attr).name)
    elif iree_codegen.OneOfKnobAttr.isinstance(attr):
        knob = iree_codegen.OneOfKnobAttr(attr)
        names.append(knob.name)
    elif isinstance(attr, ir.DictAttr):
        for i in range(len(attr)):
            _collect_knob_names(attr[i].attr, names)
    elif isinstance(attr, ir.ArrayAttr):
        for i in range(len(attr)):
            _collect_knob_names(attr[i], names)


def extract_knob_names(constraints_op) -> list[str]:
    """
    Walk the knobs dict attribute tree and extract all knob names.

    Returns:
        List of knob variable names (e.g. ["wg_0", "wg_1", "red_2",
        "mma_idx", "sg_m_cnt", "sg_n_cnt", ...]).
    """
    knobs = ir.DictAttr(constraints_op.attributes["knobs"])
    names: list[str] = []
    _collect_knob_names(knobs, names)
    return names


def _collect_one_of_domains(attr, domains: dict[str, int]):
    """Recursively walk an attribute tree collecting one-of knob domains."""
    if iree_codegen.OneOfKnobAttr.isinstance(attr):
        knob = iree_codegen.OneOfKnobAttr(attr)
        domains[knob.name] = len(knob.options)
    elif isinstance(attr, ir.DictAttr):
        for i in range(len(attr)):
            _collect_one_of_domains(attr[i].attr, domains)
    elif isinstance(attr, ir.ArrayAttr):
        for i in range(len(attr)):
            _collect_one_of_domains(attr[i], domains)


def extract_one_of_knob_domains(constraints_op) -> dict[str, int]:
    knobs = ir.DictAttr(constraints_op.attributes["knobs"])
    domains: dict[str, int] = {}
    _collect_one_of_domains(knobs, domains)
    return domains


# Mapping from pipeline name strings (after _make_module_parseable) to pipeline
# enum values for _fix_pipeline_attr.
_PIPELINE_NAME_TO_ENUM: dict = {}


def _init_pipeline_map():
    """Lazily initialize _PIPELINE_NAME_TO_ENUM from the Python enum."""
    global _PIPELINE_NAME_TO_ENUM
    if _PIPELINE_NAME_TO_ENUM:
        return
    _PIPELINE_NAME_TO_ENUM = {
        "LLVMGPUVectorDistribute": iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
        "VectorDistribute": iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
        "LLVMGPUTileAndFuse": iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
        "TileAndFuse": iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
    }


def _fix_pipeline_attr(constraints_op) -> bool:
    """
    Replace StringAttr pipeline with proper pipeline attr.

    After _make_module_parseable replaces #iree_codegen< LLVMGPUVectorDistribute>
    with "LLVMGPUVectorDistribute", the pipeline attribute on the constraints op
    becomes a StringAttr. This function converts it back to a proper
    PipelineAttr so that materialize_compilation_info works.

    Returns True if the pipeline was fixed, False if the pipeline name is unknown.
    """
    _init_pipeline_map()
    pipeline_attr = constraints_op.attributes["pipeline"]
    pipeline_str = str(pipeline_attr).strip('"')
    pipeline_enum = _PIPELINE_NAME_TO_ENUM.get(pipeline_str)
    if pipeline_enum is None:
        # Already a proper pipeline attr or unknown.
        try:
            iree_codegen.DispatchLoweringPassPipelineAttr(pipeline_attr)
            return True
        except Exception:
            return False
    constraints_op.attributes["pipeline"] = (
        iree_codegen.DispatchLoweringPassPipelineAttr.get(pipeline_enum)
    )
    return True


def _get_intrinsic_attr(
    mma: iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic,
) -> iree_gpu.MMAIntrinsicAttr | iree_gpu.VirtualMMAIntrinsicAttr:
    """Given an MMA enum value, return the corresponding intrinsic attribute."""
    if isinstance(mma, iree_gpu.MMAIntrinsic):
        return iree_gpu.MMAIntrinsicAttr.get(mma)
    return iree_gpu.VirtualMMAIntrinsicAttr.get(mma)


def _get_mma_attr(
    mma: iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic,
) -> iree_gpu.MMAAttr | iree_gpu.VirtualMMAAttr:
    """Given an MMA enum value, return the corresponding MMA attribute."""
    if isinstance(mma, iree_gpu.MMAIntrinsic):
        return iree_gpu.MMAAttr.get(mma)
    return iree_gpu.VirtualMMAAttr.get(mma)


def _layouts_match(
    layout_a: iree_gpu.GPUMMASingleSubgroupLayout,
    layout_b: iree_gpu.GPUMMASingleSubgroupLayout,
) -> bool:
    """Compare two GPUMMASingleSubgroupLayout objects for equality."""
    return (
        layout_a.element == layout_b.element
        and layout_a.thread == layout_b.thread
        and layout_a.tstrides == layout_b.tstrides
    )


def _find_compatible_qk_mma_intrinsics(
    qk_matmul: common.MatmulShapeType,
    pv_acc_layout: iree_gpu.GPUMMASingleSubgroupLayout,
    mma_intrinsics: list[iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic],
) -> list[tuple]:
    """
    Find QK-compatible MMA intrinsics whose ACC layout matches pv_acc_layout.

    Returns list of (mma_enum, mma_attr, intrinsic_attr) tuples.
    """
    # Get type-compatible intrinsics for the QK matmul.
    qk_lhs_type = common.ShapedType(
        [qk_matmul.m, qk_matmul.k], qk_matmul.lhs_type
    )
    qk_rhs_type = common.ShapedType(
        [qk_matmul.k, qk_matmul.n], qk_matmul.rhs_type
    )
    qk_acc_type = common.ShapedType(
        [qk_matmul.m, qk_matmul.n], qk_matmul.acc_type
    )
    compatible = rocm_common.get_compatible_mma_intrinsics(
        qk_lhs_type, qk_rhs_type, qk_acc_type, mma_intrinsics, allow_virtual_mma=True
    )

    results = []
    for mma_enum in compatible:
        intrinsic_attr = _get_intrinsic_attr(mma_enum)
        mma_attr = _get_mma_attr(mma_enum)
        # Check that ACC layout of this QK intrinsic matches pv_acc_layout.
        qk_acc_layout = iree_gpu.get_single_subgroup_layout(intrinsic_attr, 2)
        if _layouts_match(qk_acc_layout, pv_acc_layout):
            results.append((mma_enum, mma_attr, intrinsic_attr))
    return results


def _check_can_reuse_qk_output(
    pv_intrinsic_attr: iree_gpu.MMAAttr | iree_gpu.VirtualMMAAttr,
) -> bool:
    """
    Check if PV ACC layout matches PV LHS or PV RHS layout.
    If so, the QK output can be reused as PV input without reshuffling.
    """
    pv_acc_layout = iree_gpu.get_single_subgroup_layout(pv_intrinsic_attr, 2)
    pv_lhs_layout = iree_gpu.get_single_subgroup_layout(pv_intrinsic_attr, 0)
    pv_rhs_layout = iree_gpu.get_single_subgroup_layout(pv_intrinsic_attr, 1)
    return _layouts_match(pv_acc_layout, pv_lhs_layout) or _layouts_match(
        pv_acc_layout, pv_rhs_layout
    )


def generate_derived_configs(
    base_config: iree_codegen.CompilationInfoAttr,
    promote_operands: list[int],
    pipeline_options_search_space: rocm_common.PipelineOptionsSearchSpace,
    allowed_waves_per_eu: list[int],
) -> list[iree_codegen.CompilationInfoAttr]:
    """
    Take a base CompilationInfoAttr from materialize_compilation_info and
    produce variants with promote_operands + pipeline_options + waves_per_eu.

    materialize_compilation_info produces a base config with the correct
    lowering_config (workgroup, reduction, thread, mma_kind, subgroup_basis)
    and translation_info (pipeline, workgroup_size, subgroup_size) but without
    promote_operands, pipeline_options, or waves_per_eu.

    This function adds those tuner-specific fields to produce the final
    CompilationInfoAttr variants.
    """
    # Add promote_operands to lowering_config.
    lc_dict = iree_gpu.LoweringConfigAttr(base_config.lowering_config).attributes
    lc_dict = ir.DictAttr(lc_dict)
    new_lc_entries: dict = {}
    for i in range(len(lc_dict)):
        na = lc_dict[i]
        new_lc_entries[na.name] = na.attr
    new_lc_entries["promote_operands"] = ir.ArrayAttr.get(
        [
            ir.IntegerAttr.get(ir.IntegerType.get_signless(64), x)
            for x in promote_operands
        ]
    )
    lowering_config = iree_gpu.LoweringConfigAttr.get(
        ir.DictAttr.get(new_lc_entries)
    )

    # Get base translation_info fields.
    base_ti = iree_codegen.TranslationInfoAttr(base_config.translation_info)

    # Generate variants over pipeline_options × waves_per_eu.
    pipeline_options_list = rocm_common.generate_allowed_pipeline_options(
        pipeline_options_search_space
    )
    configs: list[iree_codegen.CompilationInfoAttr] = []
    for pipeline_options in pipeline_options_list:
        for waves_per_eu in allowed_waves_per_eu:
            config_dict = rocm_common.get_translation_info_config(
                pipeline_options, waves_per_eu
            )
            ti = iree_codegen.TranslationInfoAttr.get(
                base_ti.pass_pipeline,
                base_ti.codegen_spec,
                list(base_ti.workgroup_size),
                base_ti.subgroup_size,
                config_dict,
            )
            configs.append(
                iree_codegen.CompilationInfoAttr.get(lowering_config, ti)
            )
    return configs


def solve_smtlib_constraints(
    smtlib_str: str,
    knob_names: list[str],
    extra_constraints_fn: Optional[Callable] = None,
    max_solutions: int = 0,
) -> list[dict[str, int]]:
    """
    Parse SMT-LIB string and enumerate all satisfying models.

    Args:
        smtlib_str: SMT-LIB 2 string from smt_constraints_op_to_smtlib.
        knob_names: List of variable names to extract from each model.
        extra_constraints_fn: Optional callback(solver, vars_dict) to add
            additional constraints (e.g. num_subgroups, sg_size fixes).

    Returns:
        List of dicts mapping variable name → integer value.
    """
    solver = z3.Solver()

    # Strip SMT-LIB commands that reset/clear the solver state.
    # The SMT-LIB export from MLIR appends (reset) at the end.
    cleaned = smtlib_str.replace("(reset)", "").replace("(exit)", "")
    formulas = z3.parse_smt2_string(cleaned)
    solver.add(formulas)

    # Build lookup dict for all declared variables.
    # z3.Int(name) with the same name as a declare-const in the parsed
    # SMT-LIB refers to the same z3 constant.
    all_vars: dict[str, z3.ExprRef] = {}
    for name in knob_names:
        all_vars[name] = z3.Int(name)

    if extra_constraints_fn is not None:
        extra_constraints_fn(solver, all_vars)

    solutions = []
    knob_vars = [all_vars[name] for name in knob_names]

    while solver.check() == z3.sat:
        model = solver.model()
        assignment = {}
        for name in knob_names:
            val = model.eval(all_vars[name])
            if z3.is_int_value(val):
                assignment[name] = val.as_long()
            else:
                # Variable not constrained; skip this solution.
                break
        else:
            solutions.append(assignment)

        # Block current solution.
        solver.add(
            z3.Or(
                [v != model.eval(v, model_completion=True) for v in knob_vars]
            )
        )

        if max_solutions > 0 and len(solutions) >= max_solutions:
            break

    return solutions



def generate_compiler_contraction_solutions(
    tuner_ctx: common.TunerContext,
    gpu_target_info: iree_gpu.TargetInfo,
    contraction_dims: common.ContractionDimensions,
    matmul_size: common.ContractionSizes,
    lhs_type: common.ShapedType,
    rhs_type: common.ShapedType,
    res_type: common.ShapedType,
    num_subgroups: int = 4,
    allowed_waves_per_eu: list[int] = [2],
    pipeline_options_search_space: rocm_common.PipelineOptionsSearchSpace = rocm_common.PipelineOptionsSearchSpace(),
    iree_opt_path: Optional[str] = None,
    module_str_builder: Optional[Callable[[iree_gpu.TargetInfo], str]] = None,
    pipeline_filter: Optional[int] = None,
) -> Iterator[list[common.TuningConfiguration]]:
    """
    End-to-end flow: build MLIR → run compiler pass → extract SMT-LIB →
    solve with Z3 → materialize CompilationInfoAttr → derive variants.

    Handles both VectorDistribute and TileAndFuse pipelines in a single pass.
    The compiler's insert-smt-constraints pass generates constraints for all
    applicable pipelines, and this function processes each constraints op
    based on its pipeline type.

    Uses materialize_compilation_info to build the base CompilationInfoAttr
    from the Z3 solution, then generate_derived_configs to add
    promote_operands, pipeline_options, and waves_per_eu variants.

    Args:
        module_str_builder: Optional callable to build custom MLIR text
            (e.g. for conv ops). If None, builds a standard matmul MLIR.
        pipeline_filter: Optional DispatchLoweringPassPipeline enum value.
            If set, only process constraints ops matching this pipeline.
            If None, process all pipelines.
    """
    M, N, K = matmul_size.M, matmul_size.N, matmul_size.K

    # Step 1: Build hal.executable MLIR text.
    if module_str_builder is not None:
        module_str = module_str_builder(gpu_target_info)
    else:
        module_str = build_hal_executable_mlir_str(
            m=math.prod(M),
            n=math.prod(N),
            k=math.prod(K),
            lhs_elem=str(lhs_type.element_type),
            rhs_elem=str(rhs_type.element_type),
            res_elem=str(res_type.element_type),
            gpu_target_info=gpu_target_info,
        )

    # Step 2: Run iree-opt to insert constraints.
    output_ir = run_insert_smt_constraints(module_str, iree_opt_path)

    # Step 3: Parse module, fix pipeline attrs, process each constraints op.
    with tuner_ctx.mlir_ctx:
        parseable_str = _make_module_parseable(output_ir)
        input_module = ir.Module.parse(parseable_str)
        constraints_ops = _get_constraints_ops(input_module)

        _VD = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute

        for op in constraints_ops:
            if not _fix_pipeline_attr(op):
                continue

            # Skip ops without workgroup_size knobs (e.g., fill ops with
            # minimal fallback constraints).
            knobs = ir.DictAttr(op.attributes["knobs"])
            if "workgroup_size" not in knobs:
                continue

            # Determine pipeline type for extra constraints.
            pipeline_val = iree_codegen.DispatchLoweringPassPipelineAttr(
                op.attributes["pipeline"]
            ).value
            is_vd = pipeline_val == _VD

            # Apply pipeline filter if specified.
            if pipeline_filter is not None and pipeline_val != pipeline_filter:
                continue

            # Extract knob names from the constraints op.
            knob_names = extract_knob_names(op)
            one_of_domains = extract_one_of_knob_domains(op)

            # Get SMT-LIB from the constraints op.
            smtlib_str = _constraints_op_to_smtlib(op)

            # Step 4: Solve with Z3, applying tuner-side policy constraints.
            def extra_constraints(
                solver,
                vars_dict,
                _is_vd=is_vd,
                _num_sg=num_subgroups,
                _one_of_domains=one_of_domains,
            ):
                for name, domain_size in _one_of_domains.items():
                    solver.add(vars_dict[name] >= 0)
                    solver.add(vars_dict[name] < domain_size)
                for name, var in vars_dict.items():
                    if name.startswith("wg_") or name.startswith("red_"):
                        solver.add(var >= 1)
                        solver.add(var <= 512)
                if "sg_size" in vars_dict:
                    solver.add(
                        vars_dict["sg_size"]
                        == gpu_target_info.subgroup_size_choices[0]
                    )
                for y_name in ("wg_y", "wg_size_y"):
                    if y_name in vars_dict:
                        solver.add(vars_dict[y_name] == 1)
                for z_name in ("wg_z", "wg_size_z"):
                    if z_name in vars_dict:
                        solver.add(vars_dict[z_name] == 1)
                if _num_sg > 0:
                    if _is_vd:
                        if {"sg_m_cnt", "sg_n_cnt"} <= vars_dict.keys():
                            solver.add(vars_dict["sg_m_cnt"] >= 1)
                            solver.add(vars_dict["sg_n_cnt"] >= 1)
                            solver.add(vars_dict["sg_m_cnt"] <= 32)
                            solver.add(vars_dict["sg_n_cnt"] <= 32)
                            solver.add(
                                vars_dict["sg_m_cnt"] * vars_dict["sg_n_cnt"]
                                == _num_sg
                            )
                        for x_name in ("wg_x", "wg_size_x"):
                            if x_name in vars_dict and "sg_size" in vars_dict:
                                solver.add(
                                    vars_dict[x_name]
                                    == _num_sg * vars_dict["sg_size"]
                                )
                    else:
                        for x_name in ("wg_x", "wg_size_x"):
                            if x_name in vars_dict and "sg_size" in vars_dict:
                                solver.add(
                                    vars_dict[x_name]
                                    == _num_sg * vars_dict["sg_size"]
                                )

            solutions = solve_smtlib_constraints(
                smtlib_str, knob_names, extra_constraints
            )

            # Step 5: Materialize + derive variants for each solution.
            for values in solutions:
                try:
                    base_config = iree_codegen.materialize_compilation_info(
                        op, values
                    )
                except RuntimeError:
                    # Invalid assignment (e.g. unknown MMA shape); skip.
                    continue

                derived = generate_derived_configs(
                    base_config,
                    [0, 1],
                    pipeline_options_search_space,
                    allowed_waves_per_eu,
                )

                for config in derived:
                    yield [
                        common.TuningConfiguration(
                            name="compilation_info",
                            configuration=config,
                        )
                    ]


def generate_compiler_attention_solutions(
    tuner_ctx: common.TunerContext,
    gpu_target_info: iree_gpu.TargetInfo,
    op_info: dispatch_parser.AttentionOpInfo,
    num_subgroups: int = 4,
    allowed_waves_per_eu: list[int] = [2],
    pipeline_options_search_space: rocm_common.PipelineOptionsSearchSpace = rocm_common.PipelineOptionsSearchSpace(),
    iree_opt_path: Optional[str] = None,
) -> Iterator[list[common.TuningConfiguration]]:
    """
    End-to-end flow for attention VectorDistribute: build MLIR → run compiler
    pass → extract SMT-LIB → solve with Z3 → materialize base config →
    post-process to build QK/PV decomposition configs → derive variants.

    The compiler constraints model the PV matmul MMA shape. After solving,
    this function:
    1. Finds QK MMA intrinsics compatible with QK types whose ACC layout
       matches the PV ACC layout.
    2. Builds separate QK/PV lowering configs with their respective
       subgroup basis mappings.
    3. Determines prefetch_num_stages from PV layout matching.
    """
    # Step 1: Build attention MLIR text.
    module_str = build_hal_executable_attention_mlir_str(
        batch=math.prod(op_info.batch_sizes) if op_info.batch_sizes else 1,
        seq_len=math.prod(op_info.m_sizes),
        head_dim_k1=math.prod(op_info.k1_sizes),
        kv_seq_len=math.prod(op_info.k2_sizes),
        head_dim_n=math.prod(op_info.n_sizes),
        q_elem=str(op_info.query_type),
        k_elem=str(op_info.key_type),
        v_elem=str(op_info.value_type),
        res_elem=str(op_info.output_type),
        gpu_target_info=gpu_target_info,
    )

    # Step 2: Run iree-opt to insert constraints.
    output_ir = run_insert_smt_constraints(module_str, iree_opt_path)

    # Step 3: Parse module, fix pipeline attrs, process each constraints op.
    with tuner_ctx.mlir_ctx:
        parseable_str = _make_module_parseable(output_ir)
        input_module = ir.Module.parse(parseable_str)
        constraints_ops = _get_constraints_ops(input_module)

        for op in constraints_ops:
            if not _fix_pipeline_attr(op):
                continue

            # Only process VD attention constraints (with workgroup_size and subgroup_basis).
            knobs = ir.DictAttr(op.attributes["knobs"])
            if "workgroup_size" not in knobs:
                continue
            if "subgroup_basis" not in knobs:
                continue

            # Extract knob names and SMT-LIB.
            knob_names = extract_knob_names(op)
            smtlib_str = _constraints_op_to_smtlib(op)

            # Step 4: Solve with Z3.
            def extra_constraints(
                solver, vars_dict, _num_sg=num_subgroups
            ):
                if _num_sg > 0:
                    solver.add(
                        vars_dict["sg_m_cnt"] * vars_dict["sg_n_cnt"]
                        == _num_sg
                    )

            solutions = solve_smtlib_constraints(
                smtlib_str, knob_names, extra_constraints
            )

            # Step 5: For each solution, materialize base config and
            # build QK/PV decomposition configs.
            # Hoist loop-invariant knobs dict lookup.
            knobs = ir.DictAttr(op.attributes["knobs"])
            mma_kind_attr = knobs["mma_kind"]
            one_of_knob = iree_codegen.OneOfKnobAttr(mma_kind_attr)

            for values in solutions:
                try:
                    base_config = iree_codegen.materialize_compilation_info(
                        op, values
                    )
                except RuntimeError:
                    continue

                # Find PV MMA intrinsic from the solved mma_idx.
                mma_idx = values["mma_idx"]
                options = one_of_knob.options
                if mma_idx < 0 or mma_idx >= len(options):
                    continue
                pv_mma_option = options[mma_idx]

                # Cast to the concrete MMA type.
                if iree_gpu.MMAAttr.isinstance(pv_mma_option):
                    pv_mma_attr = iree_gpu.MMAAttr(pv_mma_option)
                elif iree_gpu.VirtualMMAAttr.isinstance(pv_mma_option):
                    pv_mma_attr = iree_gpu.VirtualMMAAttr(pv_mma_option)
                else:
                    continue

                # Verify we can get the shape (skip if unsupported).
                try:
                    _ = pv_mma_attr.mnk_shape
                except ValueError:
                    continue

                # get_single_subgroup_layout accepts MMAAttr/VirtualMMAAttr
                # directly (not just IntrinsicAttr variants).
                pv_acc_layout = iree_gpu.get_single_subgroup_layout(
                    pv_mma_attr, 2
                )

                # Check if QK output can be reused as PV input.
                can_reuse = _check_can_reuse_qk_output(pv_mma_attr)

                # Find compatible QK MMAs whose ACC layout matches PV ACC layout.
                qk_matches = _find_compatible_qk_mma_intrinsics(
                    op_info.qk_matmul,
                    pv_acc_layout,
                    gpu_target_info.mma_intrinsics,
                )

                if not qk_matches:
                    continue

                sg_m_cnt = values["sg_m_cnt"]
                sg_n_cnt = values["sg_n_cnt"]

                for _qk_mma_enum, qk_mma_attr, _qk_intrinsic_attr in qk_matches:
                    # Build subgroup basis counts.
                    subgroup_basis_counts = [1] * op_info.domain_rank
                    subgroup_basis_counts[op_info.m_dims[-1]] = sg_m_cnt
                    subgroup_basis_counts[op_info.n_dims[-1]] = sg_n_cnt

                    # Build QK basis mapping: all dims except n_dims.
                    subgroup_basis_mapping = list(range(op_info.domain_rank))
                    qk_basis_mapping = [
                        m
                        for i, m in enumerate(subgroup_basis_mapping)
                        if i not in op_info.n_dims
                    ]

                    # Build PV basis mapping: all dims except k1_dims.
                    pv_basis_mapping = [
                        m
                        for i, m in enumerate(subgroup_basis_mapping)
                        if i not in op_info.k1_dims
                    ]

                    # Build QK lowering config.
                    qk_lowering_config = common.get_lowering_config(
                        tuner_ctx=tuner_ctx,
                        mma_kind=qk_mma_attr,
                        subgroup_basis=[subgroup_basis_counts, qk_basis_mapping],
                        promote_operands=[0, 1],
                    )

                    # Build PV lowering config.
                    pv_lowering_config = common.get_lowering_config(
                        tuner_ctx=tuner_ctx,
                        mma_kind=pv_mma_attr,
                        subgroup_basis=[subgroup_basis_counts, pv_basis_mapping],
                        promote_operands=[1],
                    )

                    # Build decomposition config.
                    decomposition_config = (
                        rocm_common.get_attention_decomposition_config(
                            tuner_ctx, qk_lowering_config, pv_lowering_config
                        )
                    )

                    # Set prefetch based on layout reuse.
                    local_pipeline_opts = rocm_common.PipelineOptionsSearchSpace(
                        prefetch_num_stages=[2 if can_reuse else 0],
                        no_reduce_shared_memory_bank_conflicts=pipeline_options_search_space.no_reduce_shared_memory_bank_conflicts,
                        use_igemm_convolution=pipeline_options_search_space.use_igemm_convolution,
                    )

                    # Generate derived configs with promote_operands [0, 1, 2]
                    # and pipeline_options + waves_per_eu variants.
                    derived = generate_derived_configs(
                        base_config,
                        [0, 1, 2],
                        local_pipeline_opts,
                        allowed_waves_per_eu,
                    )

                    for config in derived:
                        yield [
                            common.TuningConfiguration(
                                name="compilation_info",
                                configuration=config,
                            ),
                            common.TuningConfiguration(
                                name="decomposition_config",
                                configuration=decomposition_config,
                            ),
                        ]


def solve_attention_constraints(
    tuner_ctx: common.TunerContext,
    gpu_target_info: iree_gpu.TargetInfo,
    batch: int,
    seq_len: int,
    head_dim_k1: int,
    kv_seq_len: int,
    head_dim_n: int,
    q_elem: str = "f16",
    k_elem: str = "f16",
    v_elem: str = "f16",
    res_elem: str = "f16",
    num_subgroups: int = 4,
    max_solutions: int = 0,
    iree_opt_path: Optional[str] = None,
) -> list[dict[str, int]]:
    """
    End-to-end flow for attention VectorDistribute constraints: build attention
    MLIR -> run compiler pass -> extract SMT-LIB -> solve with Z3.

    Returns raw knob value dicts (wg_1, wg_4, red_3, mma_idx,
    sg_m_cnt, sg_n_cnt, wg_x, sg_size, etc.).

    The 5D iteration domain is [B=d0, M=d1, K1=d2, K2=d3, N=d4].
    """
    module_str = build_hal_executable_attention_mlir_str(
        batch=batch,
        seq_len=seq_len,
        head_dim_k1=head_dim_k1,
        kv_seq_len=kv_seq_len,
        head_dim_n=head_dim_n,
        q_elem=q_elem,
        k_elem=k_elem,
        v_elem=v_elem,
        res_elem=res_elem,
        gpu_target_info=gpu_target_info,
    )

    output_ir = run_insert_smt_constraints(module_str, iree_opt_path)

    with tuner_ctx.mlir_ctx:
        smtlib_results = extract_smtlib_from_module(output_ir, tuner_ctx.mlir_ctx)

    if not smtlib_results:
        return []

    # Filter to VectorDistribute attention constraints (have subgroup_basis).
    smtlib_results = [
        (s, k)
        for s, k in smtlib_results
        if "workgroup_size" in str(k) and "subgroup_basis" in str(k)
    ]
    if not smtlib_results:
        return []

    # Attention VD knob names: d0=B, d1=M, d2=K1, d3=K2, d4=N.
    knob_names = [
        "wg_1",
        "wg_4",
        "red_3",
        "mma_idx",
        "sg_m_cnt",
        "sg_n_cnt",
        "wg_x",
        "wg_y",
        "wg_z",
        "sg_size",
    ]

    all_solutions = []
    for smtlib_str, knobs_attr in smtlib_results:

        def extra_constraints(solver, vars_dict):
            if num_subgroups > 0:
                sg_m = vars_dict["sg_m_cnt"]
                sg_n = vars_dict["sg_n_cnt"]
                solver.add(sg_m * sg_n == num_subgroups)

        solutions = solve_smtlib_constraints(
            smtlib_str, knob_names, extra_constraints, max_solutions
        )
        all_solutions.extend(solutions)

    return all_solutions
