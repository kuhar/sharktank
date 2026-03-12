# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for rocm_compiler_constraints: the compiler-side constraint integration.

Usage: python -m pytest tests/rocm/rocm_compiler_constraints_test.py
"""

import pytest
import shutil

import z3  # type: ignore

# TODO: remove after https://github.com/llvm/llvm-project/pull/117918 is resolved.
import amdsharktuner
from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, iree_gpu  # type: ignore

from iree.compiler.dialects import arith, func, linalg as linalg_dialect  # type: ignore

from amdsharktuner import (
    common,
    dispatch_parser,
)
from amdsharktuner.rocm import (
    rocm_common,
    rocm_compiler_constraints,
)

from amdsharktuner.test_utils import tuner_ctx


def build_func_with_matmul(
    module: ir.Module,
    m: int,
    n: int,
    k: int,
    lhs_type: ir.Type,
    rhs_type: ir.Type,
    res_type: ir.Type,
) -> None:
    a_type = ir.RankedTensorType.get((m, k), lhs_type)
    b_type = ir.RankedTensorType.get((k, n), rhs_type)
    c_type = ir.RankedTensorType.get((m, n), res_type)

    dim_m = ir.AffineDimExpr.get(0)
    dim_n = ir.AffineDimExpr.get(1)
    dim_k = ir.AffineDimExpr.get(2)
    a_map = ir.AffineMap.get(3, 0, [dim_m, dim_k])
    b_map = ir.AffineMap.get(3, 0, [dim_k, dim_n])
    c_map = ir.AffineMap.get(3, 0, [dim_m, dim_n])

    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(a_type, b_type, c_type)
        def named_matmul(a: ir.Value, b: ir.Value, c: ir.Value) -> None:
            matmul_op = linalg_dialect.MatmulOp(
                result_tensors=[c_type],
                inputs=[a, b],
                outputs=[c],
                indexing_maps=[a_map, b_map, c_map],
            )
            matmul_op.operation.attributes["root_op"] = iree_codegen.RootOpAttr.get(set=0)


@pytest.fixture
def gpu_target_info(tuner_ctx: common.TunerContext) -> iree_gpu.TargetInfo:
    context = tuner_ctx.mlir_ctx
    return iree_gpu.TargetInfo(
        context=context,
        arch="gfx942",
        subgroup_size_choices=[64],
        max_workgroup_sizes=[1024, 1024, 1024],
        max_thread_count_per_workgroup=1024,
        max_workgroup_memory_bytes=65536,
        workgroup_count=304,
        simds_per_workgroup=4,
        mma_intrinsics=[
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
        ],
    )


requires_iree_opt = pytest.mark.skipif(
    shutil.which("iree-opt") is None,
    reason="iree-opt not found on PATH",
)


@requires_iree_opt
def test_build_hal_executable_mlir_str(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    """Test that build_hal_executable_mlir_str produces parseable MLIR."""
    module_str = rocm_compiler_constraints.build_hal_executable_mlir_str(
        m=128,
        n=512,
        k=256,
        lhs_elem="f16",
        rhs_elem="f16",
        res_elem="f32",
        gpu_target_info=gpu_target_info,
    )
    assert "hal.executable" in module_str
    assert "linalg.matmul" in module_str
    assert "gfx942" in module_str


@requires_iree_opt
def test_run_insert_smt_constraints(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    """Test that running iree-opt inserts constraints ops."""
    module_str = rocm_compiler_constraints.build_hal_executable_mlir_str(
        m=128,
        n=512,
        k=256,
        lhs_elem="f16",
        rhs_elem="f16",
        res_elem="f32",
        gpu_target_info=gpu_target_info,
    )
    output_ir = rocm_compiler_constraints.run_insert_smt_constraints(module_str)
    assert "iree_codegen.smt.constraints" in output_ir
    assert "iree_codegen.smt.knob" in output_ir
    assert "iree_codegen.smt.assert" in output_ir


@requires_iree_opt
def test_extract_smtlib_from_module(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    """Test that SMT-LIB extraction works."""
    module_str = rocm_compiler_constraints.build_hal_executable_mlir_str(
        m=128,
        n=512,
        k=256,
        lhs_elem="f16",
        rhs_elem="f16",
        res_elem="f32",
        gpu_target_info=gpu_target_info,
    )
    output_ir = rocm_compiler_constraints.run_insert_smt_constraints(module_str)

    with tuner_ctx.mlir_ctx:
        results = rocm_compiler_constraints.extract_smtlib_from_module(
            output_ir, tuner_ctx.mlir_ctx
        )
    # Constraints ops for fill (minimal, 2 pipelines) and matmul (full, 2 pipelines).
    # Each pipeline generates its own constraint op.
    assert len(results) >= 2
    # Find VD contraction constraints (has subgroup_basis but no subgroup tiling).
    vd_results = [
        (s, k)
        for s, k in results
        if "workgroup_size" in str(k)
        and "subgroup_basis" in str(k)
        and "subgroup = [" not in str(k)
    ]
    assert len(vd_results) == 1
    smtlib_str, knobs_attr = vd_results[0]
    assert "declare-const" in smtlib_str
    assert "assert" in smtlib_str
    assert "wg_0" in smtlib_str
    assert "mma_idx" in smtlib_str

    # Find TaF contraction constraints (has both subgroup tiling + subgroup_basis).
    taf_results = [
        (s, k)
        for s, k in results
        if "workgroup_size" in str(k)
        and "subgroup = [" in str(k)
    ]
    assert len(taf_results) == 1


def test_smtlib_solving_basic() -> None:
    """Test Z3 parsing and model enumeration with a small SMT-LIB string."""
    smtlib = """
    (declare-const x Int)
    (declare-const y Int)
    (assert (>= x 1))
    (assert (<= x 3))
    (assert (>= y 1))
    (assert (<= y 2))
    (assert (= (mod (* x y) 2) 0))
    """
    solutions = rocm_compiler_constraints.solve_smtlib_constraints(
        smtlib, ["x", "y"]
    )
    assert len(solutions) > 0

    # All solutions should satisfy x*y % 2 == 0.
    for sol in solutions:
        assert (sol["x"] * sol["y"]) % 2 == 0, f"Invalid solution: {sol}"

    # Check we got all expected solutions.
    # x in [1,2,3], y in [1,2], x*y even.
    # Valid: (1,2), (2,1), (2,2), (3,2) = 4 solutions.
    expected = {(1, 2), (2, 1), (2, 2), (3, 2)}
    actual = {(s["x"], s["y"]) for s in solutions}
    assert actual == expected, f"Expected {expected}, got {actual}"


@requires_iree_opt
def test_compiler_constraint_generator_produces_solutions(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    """Test that the compiler constraint path produces at least one valid solution."""
    context = tuner_ctx.mlir_ctx
    f16 = tuner_ctx.type.f16
    f32 = tuner_ctx.type.f32

    m, n, k = 256, 512, 128
    with ir.Location.unknown(context):
        module = ir.Module.create()
        build_func_with_matmul(module, m, n, k, f16, f16, f32)

        root_ops = iree_codegen.get_tuner_root_ops(module)
        assert len(root_ops) == 1
        root_op = root_ops[0]

        parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)
        op_info = parser.get_op_info()

    configs = rocm_compiler_constraints.generate_compiler_contraction_solutions(
        tuner_ctx=tuner_ctx,
        gpu_target_info=gpu_target_info,
        contraction_dims=op_info.dims,
        matmul_size=op_info.matmul_size,
        lhs_type=op_info.lhs_type,
        rhs_type=op_info.rhs_type,
        res_type=op_info.res_type,
        pipeline_filter=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
        num_subgroups=4,
        pipeline_options_search_space=rocm_common.PipelineOptionsSearchSpace(),
    )

    solutions = list(configs)
    assert len(solutions) > 0, "Expected at least one valid solution from compiler path"

    for solution in solutions:
        assert len(solution) == 1
        config = solution[0]
        assert config.name == "compilation_info"
        assert isinstance(config.configuration, iree_codegen.CompilationInfoAttr)


@requires_iree_opt
def test_compiler_contraction_solutions_produce_valid_configs(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    """
    Verify that the compiler constraint path produces valid CompilationInfoAttr
    configs for a standard matmul (both VD and TaF pipelines).
    """
    context = tuner_ctx.mlir_ctx
    f16 = tuner_ctx.type.f16
    f32 = tuner_ctx.type.f32

    m, n, k = 256, 512, 128
    with ir.Location.unknown(context):
        module = ir.Module.create()
        build_func_with_matmul(module, m, n, k, f16, f16, f32)

        root_ops = iree_codegen.get_tuner_root_ops(module)
        root_op = root_ops[0]

        parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)
        op_info = parser.get_op_info()

    # Generate VD solutions.
    vd_configs = list(
        rocm_compiler_constraints.generate_compiler_contraction_solutions(
            tuner_ctx=tuner_ctx,
            gpu_target_info=gpu_target_info,
            contraction_dims=op_info.dims,
            matmul_size=op_info.matmul_size,
            lhs_type=op_info.lhs_type,
            rhs_type=op_info.rhs_type,
            res_type=op_info.res_type,
            pipeline_filter=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
            num_subgroups=4,
            pipeline_options_search_space=rocm_common.PipelineOptionsSearchSpace(),
        )
    )

    # Generate TaF solutions.
    taf_configs = list(
        rocm_compiler_constraints.generate_compiler_contraction_solutions(
            tuner_ctx=tuner_ctx,
            gpu_target_info=gpu_target_info,
            contraction_dims=op_info.dims,
            matmul_size=op_info.matmul_size,
            lhs_type=op_info.lhs_type,
            rhs_type=op_info.rhs_type,
            res_type=op_info.res_type,
            pipeline_filter=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
            num_subgroups=4,
            pipeline_options_search_space=rocm_common.PipelineOptionsSearchSpace(),
        )
    )

    assert len(vd_configs) > 0, "VD path produced no solutions"
    assert len(taf_configs) > 0, "TaF path produced no solutions"

    for config_list in vd_configs + taf_configs:
        assert len(config_list) == 1
        config = config_list[0]
        assert config.name == "compilation_info"
        assert isinstance(config.configuration, iree_codegen.CompilationInfoAttr)


@requires_iree_opt
def test_build_hal_executable_conv_mlir_str(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    """Test that build_hal_executable_conv_mlir_str produces parseable MLIR."""
    module_str = rocm_compiler_constraints.build_hal_executable_conv_mlir_str(
        batch=1,
        oh=8,
        ow=8,
        oc=64,
        ih=10,
        iw=10,
        ic=32,
        fh=3,
        fw=3,
        input_elem="f16",
        filter_elem="f16",
        res_elem="f32",
        gpu_target_info=gpu_target_info,
    )
    assert "hal.executable" in module_str
    assert "linalg.conv_2d_nhwc_hwcf" in module_str
    assert "gfx942" in module_str


@requires_iree_opt
def test_conv_insert_smt_constraints(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    """Test that running iree-opt inserts constraints for conv ops."""
    module_str = rocm_compiler_constraints.build_hal_executable_conv_mlir_str(
        batch=1,
        oh=8,
        ow=8,
        oc=64,
        ih=10,
        iw=10,
        ic=32,
        fh=3,
        fw=3,
        input_elem="f16",
        filter_elem="f16",
        res_elem="f32",
        gpu_target_info=gpu_target_info,
    )
    output_ir = rocm_compiler_constraints.run_insert_smt_constraints(module_str)
    assert "iree_codegen.smt.constraints" in output_ir
    assert "iree_codegen.smt.knob" in output_ir
    assert "iree_codegen.smt.assert" in output_ir

    # Conv constraints should have 7 dims and conv-specific knob names.
    with tuner_ctx.mlir_ctx:
        results = rocm_compiler_constraints.extract_smtlib_from_module(
            output_ir, tuner_ctx.mlir_ctx
        )

    # Find VD constraints (has subgroup_basis but no subgroup tiling).
    vd_results = [
        (s, k)
        for s, k in results
        if "workgroup_size" in str(k)
        and "subgroup_basis" in str(k)
        and "subgroup = [" not in str(k)
    ]
    assert len(vd_results) == 1
    smtlib_str, knobs_attr = vd_results[0]

    # Conv VD constraint should have knobs for all M/N/K dims:
    # wg_0 (batch), wg_1 (oh), wg_2 (ow), wg_3 (oc) for workgroup
    # red_4 (kh), red_5 (kw), red_6 (ic) for reduction
    assert "wg_0" in smtlib_str
    assert "wg_1" in smtlib_str
    assert "wg_2" in smtlib_str
    assert "wg_3" in smtlib_str
    assert "red_4" in smtlib_str
    assert "red_5" in smtlib_str
    assert "red_6" in smtlib_str
    assert "mma_idx" in smtlib_str


@requires_iree_opt
def test_taf_extract_smtlib(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    """Test that TileAndFuse constraint extraction works for matmul."""
    module_str = rocm_compiler_constraints.build_hal_executable_mlir_str(
        m=128,
        n=512,
        k=256,
        lhs_elem="f16",
        rhs_elem="f16",
        res_elem="f32",
        gpu_target_info=gpu_target_info,
    )
    output_ir = rocm_compiler_constraints.run_insert_smt_constraints(module_str)

    with tuner_ctx.mlir_ctx:
        results = rocm_compiler_constraints.extract_smtlib_from_module(
            output_ir, tuner_ctx.mlir_ctx
        )

    # Find TaF contraction constraints (has both subgroup tiling + subgroup_basis).
    taf_results = [
        (s, k)
        for s, k in results
        if "workgroup_size" in str(k)
        and "subgroup = [" in str(k)
    ]
    assert len(taf_results) == 1
    smtlib_str, knobs_attr = taf_results[0]
    assert "declare-const" in smtlib_str
    assert "assert" in smtlib_str
    # TaF has workgroup, reduction, subgroup tile count, and subgroup count knobs.
    assert "wg_0" in smtlib_str
    assert "wg_1" in smtlib_str
    assert "red_2" in smtlib_str
    assert "sg_0_tcnt" in smtlib_str
    assert "sg_1_tcnt" in smtlib_str
    assert "sg_m_cnt" in smtlib_str
    assert "sg_n_cnt" in smtlib_str
    assert "mma_idx" in smtlib_str


@requires_iree_opt
def test_taf_compiler_constraint_generator_produces_solutions(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    """Test that the TaF compiler constraint path produces at least one valid solution."""
    context = tuner_ctx.mlir_ctx
    f16 = tuner_ctx.type.f16
    f32 = tuner_ctx.type.f32

    m, n, k = 256, 512, 128
    with ir.Location.unknown(context):
        module = ir.Module.create()
        build_func_with_matmul(module, m, n, k, f16, f16, f32)

        root_ops = iree_codegen.get_tuner_root_ops(module)
        assert len(root_ops) == 1
        root_op = root_ops[0]

        parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)
        op_info = parser.get_op_info()

    configs = rocm_compiler_constraints.generate_compiler_contraction_solutions(
        tuner_ctx=tuner_ctx,
        gpu_target_info=gpu_target_info,
        contraction_dims=op_info.dims,
        matmul_size=op_info.matmul_size,
        lhs_type=op_info.lhs_type,
        rhs_type=op_info.rhs_type,
        res_type=op_info.res_type,
        pipeline_filter=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
        num_subgroups=4,
        pipeline_options_search_space=rocm_common.PipelineOptionsSearchSpace(),
    )

    solutions = list(configs)
    assert len(solutions) > 0, "Expected at least one valid TaF solution from compiler path"

    for solution in solutions:
        assert len(solution) == 1
        config = solution[0]
        assert config.name == "compilation_info"
        assert isinstance(config.configuration, iree_codegen.CompilationInfoAttr)


@requires_iree_opt
def test_build_hal_executable_attention_mlir_str(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    """Test that build_hal_executable_attention_mlir_str produces parseable MLIR."""
    module_str = rocm_compiler_constraints.build_hal_executable_attention_mlir_str(
        batch=2,
        seq_len=1024,
        head_dim_k1=64,
        kv_seq_len=512,
        head_dim_n=128,
        q_elem="f16",
        k_elem="f16",
        v_elem="f16",
        res_elem="f16",
        gpu_target_info=gpu_target_info,
    )
    assert "hal.executable" in module_str
    assert "iree_linalg_ext.attention" in module_str
    assert "gfx942" in module_str


@requires_iree_opt
def test_attention_insert_smt_constraints(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    """Test that running iree-opt inserts constraints for attention ops."""
    module_str = rocm_compiler_constraints.build_hal_executable_attention_mlir_str(
        batch=2,
        seq_len=1024,
        head_dim_k1=64,
        kv_seq_len=512,
        head_dim_n=128,
        q_elem="f16",
        k_elem="f16",
        v_elem="f16",
        res_elem="f16",
        gpu_target_info=gpu_target_info,
    )
    output_ir = rocm_compiler_constraints.run_insert_smt_constraints(module_str)
    assert "iree_codegen.smt.constraints" in output_ir
    assert "iree_codegen.smt.knob" in output_ir
    assert "iree_codegen.smt.assert" in output_ir

    with tuner_ctx.mlir_ctx:
        results = rocm_compiler_constraints.extract_smtlib_from_module(
            output_ir, tuner_ctx.mlir_ctx
        )

    # Find VD attention constraints (with subgroup_basis).
    vd_results = [
        (s, k)
        for s, k in results
        if "workgroup_size" in str(k) and "subgroup_basis" in str(k)
    ]
    assert len(vd_results) == 1
    smtlib_str, knobs_attr = vd_results[0]

    # Attention VD should have workgroup knobs for M (d1) and N (d4),
    # reduction knob for K2 (d3), MMA, and subgroup basis.
    assert "wg_1" in smtlib_str
    assert "wg_4" in smtlib_str
    assert "red_3" in smtlib_str
    assert "mma_idx" in smtlib_str
    assert "sg_m_cnt" in smtlib_str
    assert "sg_n_cnt" in smtlib_str


@requires_iree_opt
def test_solve_attention_constraints(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    """Test end-to-end attention constraint solving produces valid solutions."""
    solutions = rocm_compiler_constraints.solve_attention_constraints(
        tuner_ctx=tuner_ctx,
        gpu_target_info=gpu_target_info,
        batch=2,
        seq_len=1024,
        head_dim_k1=64,
        kv_seq_len=512,
        head_dim_n=128,
        num_subgroups=4,
    )

    assert len(solutions) > 0, "Expected at least one attention VD solution"

    for sol in solutions:
        # Check all expected knobs are present.
        for knob in [
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
        ]:
            assert knob in sol, f"Missing knob {knob} in solution {sol}"

        # M tile (wg_1) divides seq_len (1024).
        assert 1024 % sol["wg_1"] == 0, f"wg_1={sol['wg_1']} doesn't divide 1024"

        # N tile (wg_4) divides head_dim_n (128).
        assert 128 % sol["wg_4"] == 0, f"wg_4={sol['wg_4']} doesn't divide 128"

        # K2 reduction (red_3) divides kv_seq_len (512).
        assert 512 % sol["red_3"] == 0, f"red_3={sol['red_3']} doesn't divide 512"

        # mma_idx should be within the valid range of compatible MMAs.
        # For f16 on gfx942 with MFMA_F32_16x16x16_F16 + MFMA_F32_32x32x8_F16,
        # there are 4 compatible options (2 base + 2 virtual).
        assert 0 <= sol["mma_idx"] < 4, f"mma_idx={sol['mma_idx']} out of range [0, 4)"

        # Subgroup counts multiply to num_subgroups.
        assert (
            sol["sg_m_cnt"] * sol["sg_n_cnt"] == 4
        ), f"sg_m_cnt * sg_n_cnt != 4: {sol['sg_m_cnt']} * {sol['sg_n_cnt']}"

        # Workgroup structure: flat (wg_y == 1, wg_z == 1).
        assert sol["wg_y"] == 1
        assert sol["wg_z"] == 1

        # wg_x = sg_m_cnt * sg_n_cnt * sg_size.
        assert sol["wg_x"] == sol["sg_m_cnt"] * sol["sg_n_cnt"] * sol["sg_size"]


@requires_iree_opt
def test_conv_taf_insert_smt_constraints(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    """Test that conv + TileAndFuse generates TaF constraints with thread knobs."""
    module_str = rocm_compiler_constraints.build_hal_executable_conv_mlir_str(
        batch=1,
        oh=8,
        ow=8,
        oc=64,
        ih=10,
        iw=10,
        ic=32,
        fh=3,
        fw=3,
        input_elem="f16",
        filter_elem="f16",
        res_elem="f32",
        gpu_target_info=gpu_target_info,
    )
    output_ir = rocm_compiler_constraints.run_insert_smt_constraints(module_str)

    with tuner_ctx.mlir_ctx:
        results = rocm_compiler_constraints.extract_smtlib_from_module(
            output_ir, tuner_ctx.mlir_ctx
        )

    # Find TaF conv constraints (has both subgroup tiling + subgroup_basis).
    taf_results = [
        (s, k)
        for s, k in results
        if "workgroup_size" in str(k)
        and "subgroup = [" in str(k)
    ]
    assert len(taf_results) == 1
    smtlib_str, knobs_attr = taf_results[0]
    # Conv TaF should have 7-dim subgroup tile count and workgroup knobs.
    assert "sg_0_tcnt" in smtlib_str
    assert "sg_1_tcnt" in smtlib_str
    assert "sg_2_tcnt" in smtlib_str
    assert "sg_3_tcnt" in smtlib_str
    assert "wg_0" in smtlib_str
    assert "wg_3" in smtlib_str
    assert "red_4" in smtlib_str
    assert "sg_m_cnt" in smtlib_str
    assert "sg_n_cnt" in smtlib_str
    assert "mma_idx" in smtlib_str


@requires_iree_opt
def test_extract_knob_names(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    """Test that extract_knob_names extracts all knob variable names."""
    module_str = rocm_compiler_constraints.build_hal_executable_mlir_str(
        m=128,
        n=512,
        k=256,
        lhs_elem="f16",
        rhs_elem="f16",
        res_elem="f32",
        gpu_target_info=gpu_target_info,
    )
    output_ir = rocm_compiler_constraints.run_insert_smt_constraints(module_str)

    with tuner_ctx.mlir_ctx:
        parseable_str = rocm_compiler_constraints._make_module_parseable(output_ir)
        input_module = ir.Module.parse(parseable_str)
        constraints_ops = iree_codegen.get_smt_constraints_ops(input_module)

        # Find a VD constraint op (has subgroup_basis but no subgroup tiling).
        vd_ops = [
            op
            for op in constraints_ops
            if "workgroup_size" in str(op.attributes["knobs"])
            and "subgroup_basis" in str(op.attributes["knobs"])
            and "subgroup = [" not in str(op.attributes["knobs"])
        ]
        assert len(vd_ops) == 1
        knob_names = rocm_compiler_constraints.extract_knob_names(vd_ops[0])

        # VD matmul should have: wg_0, wg_1, red_2, mma_idx,
        # sg_m_cnt, sg_n_cnt, wg_x, wg_y, wg_z, sg_size.
        assert "wg_0" in knob_names
        assert "wg_1" in knob_names
        assert "red_2" in knob_names
        assert "mma_idx" in knob_names
        assert "sg_m_cnt" in knob_names
        assert "sg_n_cnt" in knob_names
        assert "wg_x" in knob_names
        assert "sg_size" in knob_names

        # Find a TaF constraint op (has both subgroup tiling + subgroup_basis).
        taf_ops = [
            op
            for op in constraints_ops
            if "workgroup_size" in str(op.attributes["knobs"])
            and "subgroup = [" in str(op.attributes["knobs"])
        ]
        assert len(taf_ops) == 1
        taf_knob_names = rocm_compiler_constraints.extract_knob_names(taf_ops[0])

        # TaF matmul should have subgroup tile count knobs + subgroup counts.
        assert "sg_0_tcnt" in taf_knob_names
        assert "sg_1_tcnt" in taf_knob_names
        assert "wg_0" in taf_knob_names
        assert "mma_idx" in taf_knob_names
        assert "sg_m_cnt" in taf_knob_names
        assert "sg_n_cnt" in taf_knob_names


@requires_iree_opt
def test_fix_pipeline_attr(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    """Test that _fix_pipeline_attr converts StringAttr pipeline to proper attr."""
    module_str = rocm_compiler_constraints.build_hal_executable_mlir_str(
        m=128,
        n=512,
        k=256,
        lhs_elem="f16",
        rhs_elem="f16",
        res_elem="f32",
        gpu_target_info=gpu_target_info,
    )
    output_ir = rocm_compiler_constraints.run_insert_smt_constraints(module_str)

    with tuner_ctx.mlir_ctx:
        parseable_str = rocm_compiler_constraints._make_module_parseable(output_ir)
        input_module = ir.Module.parse(parseable_str)
        constraints_ops = iree_codegen.get_smt_constraints_ops(input_module)

        # Find a constraint op with workgroup_size (full constraints, not minimal).
        op = None
        for candidate in constraints_ops:
            knobs = ir.DictAttr(candidate.attributes["knobs"])
            if "workgroup_size" in knobs:
                op = candidate
                break
        assert op is not None, "No constraint op with workgroup_size found"

        # Pipeline should already be a proper DispatchLoweringPassPipelineAttr
        # after parsing (upstream custom<PipelineAttr> round-trips correctly).
        pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr(
            op.attributes["pipeline"]
        )
        assert pipeline_attr.value in (
            iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
            iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
        )

        # _fix_pipeline_attr should still return True (no-op for correct attr).
        result = rocm_compiler_constraints._fix_pipeline_attr(op)
        assert result is True


@requires_iree_opt
def test_generate_derived_configs(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    """Test that generate_derived_configs adds promote_operands and variants."""
    module_str = rocm_compiler_constraints.build_hal_executable_mlir_str(
        m=128,
        n=512,
        k=256,
        lhs_elem="f16",
        rhs_elem="f16",
        res_elem="f32",
        gpu_target_info=gpu_target_info,
    )
    output_ir = rocm_compiler_constraints.run_insert_smt_constraints(module_str)

    with tuner_ctx.mlir_ctx:
        parseable_str = rocm_compiler_constraints._make_module_parseable(output_ir)
        input_module = ir.Module.parse(parseable_str)
        constraints_ops = iree_codegen.get_smt_constraints_ops(input_module)

        # Find and fix a VD constraints op (has subgroup_basis but no subgroup tiling).
        op = None
        for candidate in constraints_ops:
            rocm_compiler_constraints._fix_pipeline_attr(candidate)
            knobs = ir.DictAttr(candidate.attributes["knobs"])
            if "workgroup_size" in knobs and "subgroup_basis" in knobs and "subgroup" not in knobs:
                op = candidate
                break
        assert op is not None, "No VD constraints op found"

        # Extract knob names and solve.
        knob_names = rocm_compiler_constraints.extract_knob_names(op)
        smtlib_str = iree_codegen.smt_constraints_op_to_smtlib(op)
        solutions = rocm_compiler_constraints.solve_smtlib_constraints(
            smtlib_str,
            knob_names,
            lambda s, v: s.add(v["sg_m_cnt"] * v["sg_n_cnt"] == 4),
            max_solutions=1,
        )
        assert len(solutions) == 1, "Expected exactly one solution"

        # Materialize base config.
        base_config = iree_codegen.materialize_compilation_info(op, solutions[0])
        assert base_config is not None

        # Generate derived configs with promote_operands and 2 waves_per_eu options.
        derived = rocm_compiler_constraints.generate_derived_configs(
            base_config,
            promote_operands=[0, 1],
            pipeline_options_search_space=rocm_common.PipelineOptionsSearchSpace(),
            allowed_waves_per_eu=[2, 3],
        )

        # Should have len(pipeline_options) * len(waves_per_eu) variants.
        # Default PipelineOptionsSearchSpace has 1 option, so 1 * 2 = 2.
        assert len(derived) == 2, f"Expected 2 derived configs, got {len(derived)}"

        for config in derived:
            assert isinstance(config, iree_codegen.CompilationInfoAttr)
            # Verify promote_operands was added.
            lc_dict = ir.DictAttr(
                iree_gpu.LoweringConfigAttr(config.lowering_config).attributes
            )
            assert "promote_operands" in lc_dict
            # Verify translation_info has configuration dict.
            ti = iree_codegen.TranslationInfoAttr(config.translation_info)
            assert ti.configuration is not None


@requires_iree_opt
def test_generate_compiler_attention_solutions(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    """Test end-to-end attention solution generation via compiler path."""
    f16 = tuner_ctx.type.f16
    f32 = tuner_ctx.type.f32

    op_info = dispatch_parser.AttentionOpInfo(
        root_op=None,
        indexing_maps=[],
        domain_rank=5,
        batch_dims=[0],
        m_dims=[1],
        n_dims=[4],
        k1_dims=[2],
        k2_dims=[3],
        batch_sizes=[2],
        m_sizes=[1024],
        n_sizes=[128],
        k1_sizes=[64],
        k2_sizes=[512],
        query_type=f16,
        key_type=f16,
        value_type=f16,
        output_type=f16,
        transposed_q=False,
        transposed_k=True,
        transposed_v=False,
        qk_matmul=common.MatmulShapeType(
            m=1024,
            n=512,
            k=64,
            lhs_type=f16,
            rhs_type=f16,
            acc_type=f32,
        ),
        pv_matmul=common.MatmulShapeType(
            m=1024,
            n=128,
            k=512,
            lhs_type=f16,
            rhs_type=f16,
            acc_type=f16,
        ),
    )

    solutions = list(
        rocm_compiler_constraints.generate_compiler_attention_solutions(
            tuner_ctx=tuner_ctx,
            gpu_target_info=gpu_target_info,
            op_info=op_info,
            num_subgroups=4,
            pipeline_options_search_space=rocm_common.PipelineOptionsSearchSpace(),
        )
    )

    assert len(solutions) > 0, "Expected at least one attention solution from compiler path"

    for config_list in solutions:
        assert len(config_list) == 2
        assert config_list[0].name == "compilation_info"
        assert config_list[1].name == "decomposition_config"
        assert isinstance(
            config_list[0].configuration, iree_codegen.CompilationInfoAttr
        )
        assert isinstance(config_list[1].configuration, ir.DictAttr)

        # Verify decomposition_config has qk_attrs and pv_attrs.
        decomp = config_list[1].configuration
        assert "qk_attrs" in decomp
        assert "pv_attrs" in decomp

        # Verify prefetch_num_stages is set.
        compilation_info = config_list[0].configuration
        translation_info = compilation_info.translation_info
        if translation_info.configuration:
            pipeline_options = translation_info.configuration[
                common.GPU_PIPELINE_OPTIONS_KEY
            ]
            assert isinstance(
                pipeline_options.prefetch_num_stages, int
            ), "prefetch_num_stages must be explicitly set to an int"
