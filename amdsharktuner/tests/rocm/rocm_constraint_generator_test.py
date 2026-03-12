# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import shutil

# TODO: remove after https://github.com/llvm/llvm-project/pull/117918 is resolved.
import amdsharktuner
from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, iree_gpu  # type: ignore

from amdsharktuner import (
    common,
    dispatch_parser,
)
from amdsharktuner.rocm import (
    rocm_common,
    rocm_compiler_constraints,
)

from amdsharktuner.test_utils import tuner_ctx


requires_iree_opt = pytest.mark.skipif(
    shutil.which("iree-opt") is None,
    reason="iree-opt not found on PATH",
)


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
            iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
            iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
        ],
    )


@requires_iree_opt
def test_generate_attention_solutions(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    f16 = tuner_ctx.type.f16
    f32 = tuner_ctx.type.f32

    op_info = dispatch_parser.AttentionOpInfo(
        root_op=None,
        indexing_maps=[],
        domain_rank=5,
        batch_dims=[0],
        m_dims=[1],
        n_dims=[2],
        k1_dims=[3],
        k2_dims=[4],
        batch_sizes=[2],
        m_sizes=[64],
        n_sizes=[32],
        k1_sizes=[64],
        k2_sizes=[64],
        query_type=f16,
        key_type=f16,
        value_type=f16,
        output_type=f16,
        transposed_q=True,
        transposed_k=True,
        transposed_v=False,
        qk_matmul=common.MatmulShapeType(
            m=64,
            n=64,
            k=64,
            lhs_type=f16,
            rhs_type=f16,
            acc_type=f32,
        ),
        pv_matmul=common.MatmulShapeType(
            m=64,
            n=32,
            k=64,
            lhs_type=f16,
            rhs_type=f16,
            acc_type=f32,
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

    assert len(solutions) > 0, "Expected at least one valid attention tuning solution"
    for config_list in solutions:
        assert len(config_list) == 2
        assert config_list[0].name == "compilation_info"
        assert config_list[1].name == "decomposition_config"
        assert isinstance(
            config_list[0].configuration, iree_codegen.CompilationInfoAttr
        )
        assert isinstance(config_list[1].configuration, ir.DictAttr)

        # Verify that prefetch_num_stages is set based on layout matching.
        compilation_info = config_list[0].configuration
        translation_info = compilation_info.translation_info
        if translation_info.configuration:
            pipeline_options = translation_info.configuration[
                common.GPU_PIPELINE_OPTIONS_KEY
            ]
            # prefetch_num_stages should be explicitly set to an int (not None).
            assert isinstance(
                pipeline_options.prefetch_num_stages, int
            ), "prefetch_num_stages must be explicitly set to an int"
