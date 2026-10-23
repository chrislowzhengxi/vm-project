#!/usr/bin/env bash
# Reproduce the full feasibility study (~15-25 min on a 10-core laptop, ~6 GB RAM peak).
set -euo pipefail
cd "$(dirname "$0")"
B=https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v2
mkdir -p data/raw
for f in trace_data_vmtable_vmtable.csv.gz trace_data_deployments_deployments.csv.gz \
         trace_data_subscriptions_subscriptions.csv.gz schema.csv; do
  [ -s "data/raw/$f" ] || curl -sSL --retry 3 -o "data/raw/$f" "$B/$f"
done
python3 src/01_build_profile.py
python3 src/02_features.py
python3 src/03_models.py | tee results/03_log.txt
python3 src/04_policies.py | tee results/04_log.txt
