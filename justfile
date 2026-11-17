# Monte Carlo pricing of BTC options in Rust and Python.
#
# Network: only `fetch` and `live` talk to Deribit, and only to its public, read-only
# market-data API (no account, no orders). Everything else runs offline on the frozen
# snapshot in data/.

set shell := ["bash", "-uc"]

# List recipes
default:
    @just --list --unsorted

# Verify prerequisites and set up the Python environment
tools:
    #!/usr/bin/env bash
    set -euo pipefail
    for bin in cargo uv; do command -v "$bin" >/dev/null || { echo "missing: $bin"; exit 1; }; done
    uv sync --quiet
    echo "tooling ready"

# Format check, clippy, ruff (no changes)
check:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings
    uv run ruff check python
    uv run ruff format --check python

# Apply formatting and lint fixes
fix:
    cargo fmt --all
    cargo clippy --workspace --all-targets --fix --allow-dirty
    uv run ruff check --fix python
    uv run ruff format python

# Debug build
build:
    cargo build --workspace

# Release build of the Rust crates
release:
    cargo build --workspace --release

# Build the Python module into the uv environment (release mode)
py-build:
    uv run maturin develop --release --quiet -m crates/carlo-py/Cargo.toml

# Offline test suite: Rust (release, the simulations are slow in debug) and Python
test: py-build
    cargo test --workspace --release
    uv run pytest -q

# API docs; broken intra-doc links fail
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

# Regenerate the charts embedded in docs/reading-results.md (offline, fixed inputs)
docs-assets:
    uv run python python/docs_charts.py

# Full gate: what CI would enforce
ci: check test doc

# Download a new Deribit snapshot into data/ (network, read-only)
fetch:
    uv run python python/deribit.py

# Calibrate Heston to the latest snapshot: data/heston.json, out/smile.png
calibrate:
    uv run python python/calibrate.py

# Correctness checks against the snapshot (steps 2-5)
checks: py-build
    uv run python python/checks.py

# Python vs Rust timings: out/bench.md
bench-py: py-build
    uv run python python/bench.py

# Criterion benchmarks of the Rust crate; pass e.g. `--profile release-fast-build`
bench-rs *ARGS:
    cargo bench -p carlo {{ARGS}}

# Follow live Deribit tickers and report repricing latency (network, read-only)
live *ARGS:
    cargo run -p carlo-live --release -- {{ARGS}}
