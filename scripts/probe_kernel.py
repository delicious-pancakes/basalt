#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Sunny Patel
"""Round-trip one kernel at one optimisation level, in seconds.

`scripts/roundtrip_corpus.py` runs the whole corpus and takes minutes, which is
right for a control and wrong for a debugging loop. Nearly every scheduler defect
found so far was exposed by one kernel at one optimisation level, so this runs
exactly that: compile, reschedule, run both on the card, compare the bytes.

    python scripts/probe_kernel.py s_tile_matmul
    python scripts/probe_kernel.py s_tile_matmul,s_loop_double 1,3

Nothing here rebuilds the ISA database, because a scheduler change cannot move it.
"""
