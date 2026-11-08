#!/usr/bin/env python3
"""
EAHMMF Inference Engine v1 — 10-session CUDA stability test.

Runs ten independent inference processes using a 256 MiB physical VRAM
reserve, records EAHMMF statistics after each run, checks that no expert
pins remain, then runs the existing C++ invariant test suite.

No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_MODEL = (
    Path.home()
    / "Downloads"
    / "Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf"
)

PROMPTS = [
    "Explain how a hash table works, including collision handling and average-case complexity.",
    "Write a C program that implements binary search and explain its time complexity.",
    "Compare processes and threads in an operating system. Give practical examples.",
    "Explain the working of TCP three-way handshake and why each step is needed.",
    "Describe stack, queue, linked list, and their common use cases.",
    "Explain how a compiler converts source code into an executable program.",
    "Write Python code to read a CSV file, calculate column averages, and handle invalid rows.",
    "Explain database indexing, B-trees, and the trade-offs of adding indexes.",
    "Describe the transformer attention mechanism at a high level, including query, key, and value.",
    "Design a simple task scheduler and explain how it handles priorities and starvation.",
]


def parse_metrics(output: str) -> dict:
    """Extract the EAHMMF summary fields printed by --expert-stats."""
    result: dict[str, int | str | None] = {
        "alloc_failures": None,
        "alloc_retries": None,
        "migration_failures": None,
        "cpu_fallbacks": None,
        "pinned_now": None,
        "vram_experts": None,
        "ram_experts": None,
        "ssd_experts": None,
        "gpu_calls": None,
    }

    alloc = re.search(
        r"VRAM alloc:\s*failures=(\d+)\s+retries=(\d+)", output
    )
    if alloc:
        result["alloc_failures"] = int(alloc.group(1))
        result["alloc_retries"] = int(alloc.group(2))

    # Sum failure counts from each migration path, e.g. SSD->RAM ... fail=0.
    migration_lines = [
        line for line in output.splitlines()
        if "->" in line and re.search(r"\bfail=\d+", line)
    ]
    if migration_lines:
        result["migration_failures"] = sum(
            int(re.search(r"\bfail=(\d+)", line).group(1))
            for line in migration_lines
        )

    pins = re.search(r"\bpinned_now=(\d+)", output)
    if pins:
        result["pinned_now"] = int(pins.group(1))

    residency = re.search(
        r"resident:\s*VRAM=(\d+)\s+RAM=(\d+)\s+SSD=(\d+)", output
    )
    if residency:
        result["vram_experts"] = int(residency.group(1))
        result["ram_experts"] = int(residency.group(2))
        result["ssd_experts"] = int(residency.group(3))

    compute = re.search(
        r"cuda-compute:\s*gpu_calls=(\d+)\s+cpu_fallbacks=(\d+)",
        output,
    )
    if compute:
        result["gpu_calls"] = int(compute.group(1))
        result["cpu_fallbacks"] = int(compute.group(2))

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run 10 EAHMMF CUDA inference sessions and validate stability."
    )
    parser.add_argument("--engine", type=Path, default=Path("./qwen3moe"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--invariant-binary", type=Path, default=Path("./test_eahmmf"))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--vram-budget-mb", type=int, default=3000)
    parser.add_argument("--vram-reserve-mb", type=int, default=256)
    parser.add_argument("--ram-budget-mb", type=int, default=9216)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output-dir", type=Path, default=Path("./eahmmf_v1_stress"))
    parser.add_argument("--skip-invariants", action="store_true")
    args = parser.parse_args()

    engine = args.engine.expanduser().resolve()
    model = args.model.expanduser().resolve()
    invariant_binary = args.invariant_binary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not engine.is_file():
        parser.error(f"inference executable not found: {engine}")
    if not model.is_file():
        parser.error(f"model file not found: {model}")
    if args.runs < 1 or args.tokens < 1:
        parser.error("--runs and --tokens must be positive")

    print("EAHMMF Inference Engine v1 — CUDA stability test")
    print(f"Sessions: {args.runs} | generation target: {args.tokens} tokens")
    print(
        f"VRAM budget: {args.vram_budget_mb} MiB | "
        f"physical reserve: {args.vram_reserve_mb} MiB | "
        f"RAM budget: {args.ram_budget_mb} MiB"
    )
    print(f"Logs and CSV report: {output_dir}")
    print("\nNOTE: each session is a fresh inference process.\n")

    rows: list[dict] = []
    overall_ok = True

    for index in range(args.runs):
        run_no = index + 1
        stem = f"session_{run_no:02d}"
        event_log = output_dir / f"{stem}.events.log"
        terminal_log = output_dir / f"{stem}.terminal.log"
        prompt = PROMPTS[index % len(PROMPTS)]

        command = [
            str(engine),
            "-m", str(model),
            "--device", "cuda",
            "--expert-tier", "vram",
            "--cuda-compute",
            "-c", str(args.ctx),
            "-b", str(args.batch),
            "-t", str(args.threads),
            "-n", str(args.tokens),
            "--vram-budget-mb", str(args.vram_budget_mb),
            "--vram-reserve-mb", str(args.vram_reserve_mb),
            "--ram-budget-mb", str(args.ram_budget_mb),
            "-p", prompt,
            "--expert-log", str(event_log),
            "--expert-stats",
        ]

        print(f"[{run_no:02d}/{args.runs:02d}] {prompt}")
        started = time.monotonic()
        try:
            proc = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=args.timeout,
                check=False,
            )
            output = proc.stdout or ""
            terminal_log.write_text(output, encoding="utf-8")
            elapsed = time.monotonic() - started
            metrics = parse_metrics(output)

            row = {
                "session": run_no,
                "return_code": proc.returncode,
                "elapsed_seconds": round(elapsed, 2),
                "generated_tokens_target": args.tokens,
                **metrics,
                "terminal_log": str(terminal_log),
                "event_log": str(event_log),
            }
            rows.append(row)

            missing = [
                key for key in (
                    "alloc_failures",
                    "migration_failures",
                    "cpu_fallbacks",
                    "pinned_now",
                    "vram_experts",
                    "ram_experts",
                    "ssd_experts",
                )
                if metrics[key] is None
            ]
            checks = []
            if proc.returncode != 0:
                checks.append(f"process exit={proc.returncode}")
            if missing:
                checks.append("summary fields missing: " + ", ".join(missing))
            if metrics["alloc_failures"] not in (None, 0):
                checks.append(f"allocation failures={metrics['alloc_failures']}")
            if metrics["migration_failures"] not in (None, 0):
                checks.append(f"migration failures={metrics['migration_failures']}")
            if metrics["cpu_fallbacks"] not in (None, 0):
                checks.append(f"CPU fallbacks={metrics['cpu_fallbacks']}")
            if metrics["pinned_now"] not in (None, 0):
                checks.append(f"expert pins remain={metrics['pinned_now']}")

            if checks:
                overall_ok = False
                print("  FAIL: " + " | ".join(checks))
                print(f"  Full output: {terminal_log}")
            else:
                print(
                    "  PASS | "
                    f"alloc_fail={metrics['alloc_failures']} "
                    f"mig_fail={metrics['migration_failures']} "
                    f"CPU_fallback={metrics['cpu_fallbacks']} "
                    f"pins={metrics['pinned_now']} | "
                    f"residency V/R/S="
                    f"{metrics['vram_experts']}/"
                    f"{metrics['ram_experts']}/"
                    f"{metrics['ssd_experts']} | "
                    f"{elapsed:.1f}s"
                )

        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            terminal_log.write_text(output, encoding="utf-8")
            rows.append({
                "session": run_no,
                "return_code": "TIMEOUT",
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "generated_tokens_target": args.tokens,
                "alloc_failures": None,
                "alloc_retries": None,
                "migration_failures": None,
                "cpu_fallbacks": None,
                "pinned_now": None,
                "vram_experts": None,
                "ram_experts": None,
                "ssd_experts": None,
                "gpu_calls": None,
                "terminal_log": str(terminal_log),
                "event_log": str(event_log),
            })
            overall_ok = False
            print(f"  FAIL: timed out after {args.timeout}s; see {terminal_log}")

    report = output_dir / "session_summary.csv"
    if rows:
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
        with report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nPer-session report: {report}")

    # The existing C++ test executable runs the project's invariant checks.
    # This is the repository's invariant suite, not an introspection call into
    # the already-exited inference processes.
    if not args.skip_invariants:
        if not invariant_binary.is_file():
            overall_ok = False
            print(f"\nINVARIANT SUITE: NOT RUN — binary not found: {invariant_binary}")
        else:
            print("\nRunning C++ EAHMMF invariant/test suite...")
            inv_log = output_dir / "invariant_suite.log"
            try:
                inv = subprocess.run(
                    [str(invariant_binary)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    timeout=args.timeout,
                    check=False,
                )
                inv_log.write_text(inv.stdout or "", encoding="utf-8")
                print((inv.stdout or "").strip())
                if inv.returncode != 0:
                    overall_ok = False
                    print(f"INVARIANT SUITE: FAIL (exit {inv.returncode})")
                else:
                    print("INVARIANT SUITE: PASS")
            except subprocess.TimeoutExpired:
                overall_ok = False
                print(f"INVARIANT SUITE: TIMEOUT; see {inv_log}")

    print("\n" + ("STRESS TEST PASSED" if overall_ok else "STRESS TEST FAILED"))
    print(f"Artifacts: {output_dir}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
