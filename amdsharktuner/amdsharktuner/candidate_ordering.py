import random
import logging
import csv
from typing import Optional, Any
from dataclasses import dataclass
from pathlib import Path
from enum import Enum


class CandidateOrderKind(str, Enum):
    no_sort = "no-sort"
    shuffle = "shuffle"
    heuristic = "heuristic"


def reorder_candidates(
    count: int,
    strategy: CandidateOrderKind,
) -> list[int]:
    """
    Returns a list of indices representing the new order for `count` candidates.
    Example: count=3, shuffle -> [1, 0, 2]
    """
    logging.debug(f"Selected candidate ordering strategy: {strategy}")

    if count == 0:
        return []

    original_order = list(range(count))

    match strategy:
        case CandidateOrderKind.no_sort:
            return original_order
        case CandidateOrderKind.shuffle:
            indices = list(range(count))
            random.shuffle(indices)
            return indices
        case CandidateOrderKind.heuristic:
            logging.warning(
                "Heuristic candidate ordering is no longer supported. "
                "Falling back to no-sort."
            )
            return original_order
        case _:
            assert False


@dataclass
class TuningRecord:
    """
    Records a candidate's tuning results.

    Used to analyze the candidate search space and to evaluate the
    effectiveness of candidate ordering.
    """

    gen_id: int  # Original index from candidate generation.
    candidate_id: int  # Index in candidate_trackers after reordering.
    to_compile: bool = False
    compile_status: bool = False
    to_benchmark: bool = False
    benchmark_device_id: Optional[str] = None
    benchmark_queue_position: Optional[int] = None
    benchmark_status: bool = False
    baseline_benchmark_time_us: Optional[float] = None
    benchmark_time_us: Optional[float] = None
    benchmark_speedup: Optional[float] = None
    benchmark_rank_order: Optional[int] = None


def build_tuning_records_from_order(
    sorted_order: list[int],
) -> list[TuningRecord]:
    tuning_records: list[TuningRecord] = []
    # Insert baseline entry (always candidate_id = 0, gen_id = 0).
    tuning_records.append(TuningRecord(gen_id=0, candidate_id=0))
    for sorted_position, original_gen_index in enumerate(sorted_order, start=1):
        tr = TuningRecord(
            gen_id=original_gen_index
            + 1,  # Shift by 1 to reserve gen_id=0 for baseline.
            candidate_id=sorted_position,
        )
        tuning_records.append(tr)

    return tuning_records


def flatten_records(
    tuning_records: list[TuningRecord],
) -> list[dict[str, Any]]:
    """
    Flatten a list of `TuningRecord` objects into CSV rows.

    - Each record becomes one CSV row.
    - Top-level attributes (e.g., `gen_id`, `benchmark_time_us`) appear as individual columns.
    """
    rows = []
    for tuning_record in tuning_records:
        # Drop the baseline entry.
        if tuning_record.candidate_id == 0:
            continue
        row = {}
        for attr, val in vars(tuning_record).items():
            row[attr] = val
        rows.append(row)

    return rows


def export_record_to_csv(tuning_records: list[TuningRecord], dest_file: Path) -> None:
    assert tuning_records

    rows = flatten_records(tuning_records)
    headers = list(rows[0].keys())

    with open(dest_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
