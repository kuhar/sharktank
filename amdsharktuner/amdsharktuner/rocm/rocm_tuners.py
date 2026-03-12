# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from typing import Iterator

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, iree_gpu, linalg  # type: ignore

from .. import common, dispatch_parser, spec_builder, tuner_base
from . import rocm_compiler_constraints, rocm_parsers


class ROCmContractionVectorDistributeTuner(
    tuner_base.DispatchTuner, dispatch_parser.ContractionOpInterfaceParser
):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

    @classmethod
    def supports_root_op(cls, root_op: ir.Operation) -> bool:
        if not linalg.isa_contraction_op(root_op):
            return False

        # Check if contraction has valid dimensions.
        contraction_dims = linalg.infer_contraction_dimensions(root_op)
        if not contraction_dims:
            logging.warning("No contraction dimensions found for operation")
            return False

        if not contraction_dims.m or not contraction_dims.n or not contraction_dims.k:
            logging.warning(
                f"Contraction operation with dimensions M={list(contraction_dims.m)}, "
                f"N={list(contraction_dims.n)}, K={list(contraction_dims.k)} "
                f"is not supported by the tuner yet"
            )
            return False

        return True

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        op_info = self.get_op_info()
        return rocm_compiler_constraints.generate_compiler_contraction_solutions(
            tuner_ctx=tuner_context,
            gpu_target_info=gpu_target_info,
            contraction_dims=op_info.dims,
            matmul_size=op_info.matmul_size,
            lhs_type=op_info.lhs_type,
            rhs_type=op_info.rhs_type,
            res_type=op_info.res_type,
            pipeline_filter=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
            **pipeline_constraint_options,
        )

    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        builder = spec_builder.ContractionSpecBuilder(self.get_op_info())
        return builder.build_td_spec(self._tuner_ctx, config_list)

    @classmethod
    def get_dispatch_kind(cls) -> common.DispatchKind:
        return common.DispatchKind.contraction


class ROCmContractionTileAndFuseTuner(
    tuner_base.DispatchTuner, dispatch_parser.ContractionOpInterfaceParser
):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

    @classmethod
    def supports_root_op(cls, root_op: ir.Operation) -> bool:
        if not linalg.isa_contraction_op(root_op):
            return False

        # Check if contraction has valid dimensions.
        contraction_dims = linalg.infer_contraction_dimensions(root_op)
        if not contraction_dims:
            logging.warning("No contraction dimensions found for operation")
            return False

        if not contraction_dims.m or not contraction_dims.n or not contraction_dims.k:
            logging.warning(
                f"Contraction operation with dimensions M={list(contraction_dims.m)}, "
                f"N={list(contraction_dims.n)}, K={list(contraction_dims.k)} "
                f"is not supported by the tuner yet"
            )
            return False

        return True

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        op_info = self.get_op_info()
        return rocm_compiler_constraints.generate_compiler_contraction_solutions(
            tuner_ctx=tuner_context,
            gpu_target_info=gpu_target_info,
            contraction_dims=op_info.dims,
            matmul_size=op_info.matmul_size,
            lhs_type=op_info.lhs_type,
            rhs_type=op_info.rhs_type,
            res_type=op_info.res_type,
            pipeline_filter=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
            **pipeline_constraint_options,
        )

    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        builder = spec_builder.ContractionSpecBuilder(self.get_op_info())
        return builder.build_td_spec(self._tuner_ctx, config_list)

    @classmethod
    def get_dispatch_kind(cls) -> common.DispatchKind:
        return common.DispatchKind.contraction


class ROCmConvolutionVectorDistributeTuner(
    tuner_base.DispatchTuner, rocm_parsers.InnerMNKConvolutionParser
):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

    @classmethod
    def supports_root_op(cls, root_op: ir.Operation) -> bool:
        if not linalg.isa_convolution_op(root_op):
            return False
        convolution_dims = linalg.infer_convolution_dimensions(root_op)
        if not convolution_dims:
            return False
        # Only allow 'nhwc_hwcf' convs.
        return (
            list(convolution_dims.batch) == [0]
            and list(convolution_dims.output_image) == [1, 2]
            and list(convolution_dims.output_channel) == [3]
            and list(convolution_dims.filter_loop) == [4, 5]
            and list(convolution_dims.input_channel) == [6]
            and list(convolution_dims.depth) == []
        )

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        op_info = self.get_op_info()
        return rocm_compiler_constraints.generate_compiler_contraction_solutions(
            tuner_ctx=tuner_context,
            gpu_target_info=gpu_target_info,
            contraction_dims=op_info.dims,
            matmul_size=op_info.matmul_size,
            lhs_type=op_info.lhs_type,
            rhs_type=op_info.rhs_type,
            res_type=op_info.res_type,
            module_str_builder=self._build_conv_mlir,
            pipeline_filter=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
            **pipeline_constraint_options,
        )

    def _build_conv_mlir(
        self, gpu_target_info: iree_gpu.TargetInfo
    ) -> str:
        """Build conv MLIR from the op_info shapes."""
        info = self.get_op_info()
        input_shape = info.lhs_type.shape  # [batch, ih, iw, ic]
        filter_shape = info.rhs_type.shape  # [fh, fw, ic, oc]
        output_shape = info.res_type.shape  # [batch, oh, ow, oc]
        return rocm_compiler_constraints.build_hal_executable_conv_mlir_str(
            batch=output_shape[0],
            oh=output_shape[1],
            ow=output_shape[2],
            oc=output_shape[3],
            ih=input_shape[1],
            iw=input_shape[2],
            ic=input_shape[3],
            fh=filter_shape[0],
            fw=filter_shape[1],
            input_elem=str(info.lhs_type.element_type),
            filter_elem=str(info.rhs_type.element_type),
            res_elem=str(info.res_type.element_type),
            gpu_target_info=gpu_target_info,
        )

    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        builder = spec_builder.ConvolutionSpecBuilder(self.get_op_info())
        return builder.build_td_spec(self._tuner_ctx, config_list)

    @classmethod
    def get_dispatch_kind(cls) -> common.DispatchKind:
        return common.DispatchKind.conv


class ROCmConvolutionTileAndFuseTuner(
    tuner_base.DispatchTuner, rocm_parsers.IGEMMConvolutionParser
):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

    @classmethod
    def supports_root_op(cls, root_op: ir.Operation) -> bool:
        if not linalg.isa_convolution_op(root_op):
            return False
        convolution_dims = linalg.infer_convolution_dimensions(root_op)
        if not convolution_dims:
            return False
        # Support all 2D convolutions (no depth dimension) for IGEMM.
        return list(convolution_dims.depth) == []

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        op_info = self.get_op_info()
        return rocm_compiler_constraints.generate_compiler_contraction_solutions(
            tuner_ctx=tuner_context,
            gpu_target_info=gpu_target_info,
            contraction_dims=op_info.dims,
            matmul_size=op_info.matmul_size,
            lhs_type=op_info.lhs_type,
            rhs_type=op_info.rhs_type,
            res_type=op_info.res_type,
            module_str_builder=self._build_conv_mlir,
            pipeline_filter=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
            **pipeline_constraint_options,
        )

    def _build_conv_mlir(
        self, gpu_target_info: iree_gpu.TargetInfo
    ) -> str:
        """Build conv MLIR from the op_info shapes."""
        info = self.get_op_info()
        input_shape = info.lhs_type.shape  # [batch, ih, iw, ic]
        filter_shape = info.rhs_type.shape  # [fh, fw, ic, oc]
        output_shape = info.res_type.shape  # [batch, oh, ow, oc]
        return rocm_compiler_constraints.build_hal_executable_conv_mlir_str(
            batch=output_shape[0],
            oh=output_shape[1],
            ow=output_shape[2],
            oc=output_shape[3],
            ih=input_shape[1],
            iw=input_shape[2],
            ic=input_shape[3],
            fh=filter_shape[0],
            fw=filter_shape[1],
            input_elem=str(info.lhs_type.element_type),
            filter_elem=str(info.rhs_type.element_type),
            res_elem=str(info.res_type.element_type),
            gpu_target_info=gpu_target_info,
        )

    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        builder = spec_builder.ConvolutionSpecBuilder(self.get_op_info())
        return builder.build_td_spec(self._tuner_ctx, config_list)

    @classmethod
    def get_dispatch_kind(cls) -> common.DispatchKind:
        return common.DispatchKind.conv


class ROCmAttentionVectorDistributeTuner(
    tuner_base.DispatchTuner, dispatch_parser.AttentionOpInterfaceParser
):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

    @classmethod
    def supports_root_op(cls, root_op: ir.Operation) -> bool:
        return iree_codegen.isa_attention_op(root_op)

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        return rocm_compiler_constraints.generate_compiler_attention_solutions(
            tuner_ctx=tuner_context,
            gpu_target_info=gpu_target_info,
            op_info=self.get_op_info(),
            **pipeline_constraint_options,
        )

    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        builder = spec_builder.AttentionSpecBuilder(self.get_op_info())
        return builder.build_td_spec(self._tuner_ctx, config_list)

    @classmethod
    def get_dispatch_kind(cls) -> common.DispatchKind:
        return common.DispatchKind.attention


def get_tuners_for_pipeline(
    codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline,
) -> list[type[tuner_base.DispatchTuner]]:
    """Get ROCm tuners for the given codegen pipeline."""
    if (
        codegen_pipeline
        == iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    ):
        return [
            ROCmContractionVectorDistributeTuner,
            ROCmConvolutionVectorDistributeTuner,
            ROCmAttentionVectorDistributeTuner,
        ]

    if codegen_pipeline == iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse:
        return [
            ROCmContractionTileAndFuseTuner,
            ROCmConvolutionTileAndFuseTuner,
        ]

    return []
