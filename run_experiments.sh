# Script used for training PoserNet on different data, with different configurations.

python train.py --config-file experiment_configs/small_graphs_normal_000000.yaml
python train.py --config-file experiment_configs/small_graphs_normal_000001.yaml
python train.py --config-file experiment_configs/small_graphs_normal_000002.yaml

python train.py --config-file experiment_configs/large_graphs_normal_000000.yaml
python train.py --config-file experiment_configs/large_graphs_normal_000001.yaml
python train.py --config-file experiment_configs/large_graphs_normal_000002.yaml

python train.py --config-file experiment_configs/small_graphs_loo_000000.yaml
python train.py --config-file experiment_configs/small_graphs_loo_000001.yaml
python train.py --config-file experiment_configs/small_graphs_loo_000002.yaml
python train.py --config-file experiment_configs/small_graphs_loo_000003.yaml
python train.py --config-file experiment_configs/small_graphs_loo_000004.yaml
python train.py --config-file experiment_configs/small_graphs_loo_000005.yaml
python train.py --config-file experiment_configs/small_graphs_loo_000006.yaml
