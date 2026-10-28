#!/usr/bin/env bash
set -e

clear

cd "$(dirname "$0")"
export PYTHONPATH=".."

echo ""
echo ""
echo "--- Basic Env Benchmark ---"
python3 -m rl.src.main --env basic --agent ql
python3 -m rl.src.main --env basic --agent dumb --action wait
python3 -m rl.src.main --env basic --agent dumb --action call
python3 -m rl.src.main --env basic --agent dumb --action discount
python3 -m rl.src.main --env basic --agent random

echo ""
echo "--- Real Env Benchmark ---"
python3 -m rl.src.main --env real --agent ql
python3 -m rl.src.main --env real --agent dumb --action wait
python3 -m rl.src.main --env real --agent dumb --action call
python3 -m rl.src.main --env real --agent dumb --action discount
python3 -m rl.src.main --env real --agent random

echo ""
echo "--- Learned Policy (Basic Env) ---"
python3 -m rl.src.main --env basic --agent ql --show-policy

echo ""
echo "--- Learned Policy (Real Env) ---"
python3 -m rl.src.main --env real --agent ql --show-policy

echo ""
echo ""