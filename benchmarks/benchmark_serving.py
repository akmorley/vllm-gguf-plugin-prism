# SPDX-License-Identifier: Apache-2.0
"""Launch a local Prism server and benchmark its streaming completions."""

import argparse
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

CLI = [sys.executable, "-m", "vllm.entrypoints.cli.main"]
DEFAULT_MODEL = "/home/amorley/vllm/models/Ternary-Bonsai-2-27B-PQ2_0.gguf"
DEFAULT_TOKENIZER = "/home/amorley/vllm/models/qwen38-tokenizer"


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def gpu_info(gpu):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=uuid,name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=5,
    )
    uuid, name, used, total = next(csv.reader([output.strip()]))
    return {
        "uuid": uuid.strip(),
        "name": name.strip(),
        "used_mib": int(used),
        "total_mib": int(total),
    }


class MemorySampler:
    """Sample total device memory; includes server reservations and other processes."""

    def __init__(self, gpu):
        self.gpu = gpu
        self.samples = []
        self.errors = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            try:
                self.samples.append(
                    {"time": time.time(), "used_mib": gpu_info(self.gpu)["used_mib"]}
                )
            except (subprocess.SubprocessError, ValueError) as error:
                self.errors.append(str(error))
            self.stop.wait(0.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()

    def result(self):
        return {
            "peak_device_memory_mib": max(
                (s["used_mib"] for s in self.samples), default=None
            ),
            "memory_sample_interval_s": 0.5,
            "memory_samples": self.samples,
            "memory_sampling_errors": self.errors,
        }


def wait_ready(server, url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError("Server exited during startup; inspect server.log")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError("Server startup timed out; inspect server.log")


def shutdown(server):
    if server is None:
        return
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


def handle_stop(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--gpu", default="0", help="Physical nvidia-smi index or GPU UUID"
    )
    parser.add_argument("--port", type=positive, default=8097)
    parser.add_argument(
        "--input-lengths", type=positive, nargs="+", default=[128, 1024, 3072]
    )
    parser.add_argument("--concurrency", type=positive, nargs="+", default=[1, 4, 8])
    parser.add_argument("--output-length", type=positive, default=128)
    parser.add_argument("--requests", type=positive, default=32)
    parser.add_argument(
        "--warmups",
        type=positive,
        default=2,
        help="Minimum warmup requests; raised to workload concurrency when needed",
    )
    parser.add_argument("--repeats", type=positive, default=1)
    parser.add_argument("--max-model-len", type=positive, default=4096)
    parser.add_argument(
        "--max-num-seqs",
        type=positive,
        help="Server sequence limit; defaults to the largest workload concurrency",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs for comparison",
    )
    parser.add_argument("--startup-timeout", type=positive, default=900)
    parser.add_argument("--workload-timeout", type=positive, default=1800)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("benchmarks/results")
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    if max(args.input_lengths) + args.output_length > args.max_model_len:
        parser.error("input length + output length exceeds --max-model-len")
    if args.max_num_seqs is None:
        args.max_num_seqs = max(args.concurrency)
    if args.max_num_seqs < max(args.concurrency):
        parser.error("--max-num-seqs must be at least the largest concurrency")
    if args.requests < max(args.concurrency):
        parser.error("--requests must be at least the largest concurrency")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    for path in (args.model, args.tokenizer):
        if not Path(path).exists():
            parser.error(f"Path does not exist: {path}")
    result_dir = args.result_dir.resolve()
    gpu = gpu_info(args.gpu)
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID")
    url = f"http://127.0.0.1:{args.port}"
    # Refuse to benchmark an unrelated server if the requested port is occupied.

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    result_dir.mkdir(parents=True, exist_ok=False)
    command = CLI + [
        "serve",
        args.model,
        "--tokenizer",
        args.tokenizer,
        "--served-model-name",
        "prism-27b",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        "1",
        "--language-model-only",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--no-enable-prefix-caching",
        "--seed",
        "0",
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    metadata = {
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "gpu": gpu,
        "python": sys.executable,
        "versions": {
            package: version(package)
            for package in ("vllm", "torch", "triton", "vllm-gguf-plugin")
        },
        "server_command": command,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (result_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    server = None
    startup_memory = None
    rows = []
    print(f"Results: {result_dir}", flush=True)
    try:
        with (
            (result_dir / "server.log").open("w") as log,
            MemorySampler(gpu["uuid"]) as startup_memory,
        ):
            server = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            print("Starting server; progress is in server.log", flush=True)
            wait_ready(server, url, args.startup_timeout)
        (result_dir / "startup-memory.json").write_text(
            json.dumps(startup_memory.result(), indent=2)
        )
        for length in args.input_lengths:
            for concurrency in args.concurrency:
                for repeat in range(args.repeats):
                    name = (
                        f"input{length}-output{args.output_length}"
                        f"-c{concurrency}-r{repeat + 1}"
                    )
                    bench = CLI + [
                        "bench",
                        "serve",
                        "--backend",
                        "vllm",
                        "--base-url",
                        url,
                        "--endpoint",
                        "/v1/completions",
                        "--model",
                        "prism-27b",
                        "--tokenizer",
                        args.tokenizer,
                        "--dataset-name",
                        "random",
                        "--random-input-len",
                        str(length),
                        "--random-output-len",
                        str(args.output_length),
                        "--random-range-ratio",
                        "0",
                        "--num-prompts",
                        str(args.requests),
                        "--max-concurrency",
                        str(concurrency),
                        "--request-rate",
                        "inf",
                        "--num-warmups",
                        str(max(args.warmups, concurrency)),
                        "--ignore-eos",
                        "--seed",
                        "0",
                        "--percentile-metrics",
                        "ttft,tpot,itl,e2el",
                        "--metric-percentiles",
                        "50,95,99",
                        "--save-result",
                        "--save-detailed",
                        "--result-dir",
                        str(result_dir),
                        "--result-filename",
                        name + ".json",
                        "--disable-tqdm",
                    ]
                    (result_dir / (name + "-command.json")).write_text(
                        json.dumps(bench, indent=2)
                    )
                    print(f"Running {name}", flush=True)
                    with (
                        MemorySampler(gpu["uuid"]) as memory,
                        (result_dir / (name + ".log")).open("w") as log,
                    ):
                        subprocess.run(
                            bench,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=args.workload_timeout,
                        )
                    data = json.loads((result_dir / (name + ".json")).read_text())
                    (result_dir / (name + "-memory.json")).write_text(
                        json.dumps(memory.result(), indent=2)
                    )
                    if data["completed"] != args.requests:
                        raise RuntimeError(
                            f"{name}: only {data['completed']}/{args.requests}"
                            " requests succeeded"
                        )
                    row = {
                        "workload": name,
                        "input_tokens": length,
                        "output_tokens": args.output_length,
                        "concurrency": concurrency,
                        "repeat": repeat + 1,
                        "peak_device_memory_mib": memory.result()[
                            "peak_device_memory_mib"
                        ],
                    }
                    for key in (
                        "completed",
                        "duration",
                        "request_throughput",
                        "output_throughput",
                        "total_token_throughput",
                        "mean_ttft_ms",
                        "p95_ttft_ms",
                        "mean_tpot_ms",
                        "p95_tpot_ms",
                        "mean_e2el_ms",
                        "p95_e2el_ms",
                    ):
                        row[key] = data.get(key)
                    rows.append(row)
                    with (result_dir / "summary.csv").open("w", newline="") as output:
                        writer = csv.DictWriter(output, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerows(rows)
                    (result_dir / "summary.json").write_text(json.dumps(rows, indent=2))
                    print(
                        f"{row['output_throughput']:.2f} output tok/s; "
                        f"mean TTFT {row['mean_ttft_ms']:.1f} ms",
                        flush=True,
                    )
    finally:
        shutdown(server)
        if startup_memory is not None:
            (result_dir / "startup-memory.json").write_text(
                json.dumps(startup_memory.result(), indent=2)
            )
    print(f"Complete: {result_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
