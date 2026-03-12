# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Iterator
from abc import abstractmethod

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_gpu  # type: ignore

from . import common, dispatch_parser


class DispatchTuner(dispatch_parser.DispatchParser):
    @classmethod
    @abstractmethod
    def supports_root_op(cls, root_op: ir.Operation) -> bool:
        """Check if this tuner can handle the given root operation."""
        pass

    @abstractmethod
    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        """
        Generates a transform dialect spec from a list of TuningConfiguration objects.

        Each TuningConfiguration specifies a name (e.g., "compilation_info") and
        its corresponding MLIR attribute (e.g., CompilationInfoAttr) to be applied
        to the dispatch root operation.
        """
        pass

    @abstractmethod
    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        """Generate a sequence of tuning configuration entries."""
        pass

    @classmethod
    @abstractmethod
    def get_dispatch_kind(cls) -> common.DispatchKind:
        """Returns dispatch kind"""
        pass
