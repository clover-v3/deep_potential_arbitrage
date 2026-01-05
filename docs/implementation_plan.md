# Implementation Plan - Cluster Trading System

# Goal Description
Build a modular, end-to-end differentiable pairs trading system based on clustering.

## Proposed Changes

### Core Logic [COMPLETED]
#### [MODIFY] [data_engine.py](file:///Users/coco/docker/ubuntu/home/jq/deep_potential_arbitrage/src/cluster_trading/data_engine.py)
- [x] Implement `load_wide_data`.

#### [MODIFY] [system.py](file:///Users/coco/docker/ubuntu/home/jq/deep_potential_arbitrage/src/cluster_trading/system.py)
- [x] **Portfolio**: Independent weighting.
- [x] **Wiring**: Pass `similarity_threshold` and wire `gate` through system.

#### [MODIFY] [clustering.py](file:///Users/coco/docker/ubuntu/home/jq/deep_potential_arbitrage/src/cluster_trading/clustering.py)
- [x] **Outlier Gate**: Implement `Sigmoid(MaxSim - Thrust)`.

#### [MODIFY] [signal.py](file:///Users/coco/docker/ubuntu/home/jq/deep_potential_arbitrage/src/cluster_trading/signal.py)
- [x] **Clean Consensus**: Use `gated_assignments`.
- [x] **Forced Exit**: Mask signal with `gate`.
- [x] **Activation**: Band-Pass Filter.

#### [MODIFY] [train.py](file:///Users/coco/docker/ubuntu/home/jq/deep_potential_arbitrage/src/cluster_trading/train.py)
- [x] **CLI**: Add `--stop_threshold`, `--similarity_threshold`, `--leverage_penalty`.

### Rolling Training & Automation [COMPLETED]
#### [NEW] [backtest.py](file:///Users/coco/docker/ubuntu/home/jq/deep_potential_arbitrage/src/cluster_trading/backtest.py)
- [x] Ensure it can run inference given a model and date range.

#### [NEW] [run_rolling.py](file:///Users/coco/docker/ubuntu/home/jq/deep_potential_arbitrage/src/cluster_trading/run_rolling.py)
- [x] Implement `RollingTrainer` class.
- [x] Logic: Train (size N) -> Test (size M) -> Slide (size M).
- [x] Stitch out-of-sample returns.

#### [NEW] [run_grid_search.py](file:///Users/coco/docker/ubuntu/home/jq/deep_potential_arbitrage/src/cluster_trading/run_grid_search.py)
- [x] Iterate over `window`, `n_clusters`, `similarity_threshold`.
- [x] Sequential execution loop.
- [x] Save results to `grid_search_results.csv`.

### Documentation [COMPLETED]
#### [MODIFY] [design_doc.md](file:///Users/coco/docker/ubuntu/home/jq/deep_potential_arbitrage/docs/design_doc.md)
- [x] Add `JITAgeGating` details.

## Future Work
### Differentiable TTL Layer
- [ ] Implement `JITAgeGating` module.
