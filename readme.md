# Contrasting Estimation of Pattern Prototypes for Anomaly Detection in Urban Crowd Flow (ProtoDetect)

This repository is the official implementation of ProtoDetect.

## Dataset

- All the dataset used in the paper are public available

    - NYC Open Data https://data.cityofnewyork.us/

    - Chicago Data Portal https://data.cityofchicago.org/

- A sample dataset is provided in this repository for test

    - `dataset/NYC_dynamic_201410_minmax.npy` the crowd flow dynamics of NYC in October 2014

    - `dataset/NYC_graph_dict.json` the dictionary of subgraphs and adjacency matrices

## Test

1. Install requirements

2. Train

    ```bash
    # train the local and global ST-encoders
    python Extractor.py --mode local
    python Extractor.py --mode global

    # train ProtoDetect
    python ProtoDetect.py --mode train
    ```

    These scripts train the models on NYC dataset and save their weights in `training_cache/` directory.

3. Test

    ```bash
    python ProtoDetect.py --mode eval
    ```
    This script outputs the flattened anomaly scores `anomaly_score.npy`.

## Citation
```bib
@article{wang2024contrasting,
  title={Contrasting Estimation of Pattern Prototypes for Anomaly Detection in Urban Crowd Flow},
  author={Wang, Yupeng and Luo, Xiling and Zhou, Zequan},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2024},
  volume={},
  number={},
  pages={1-15},
  publisher={IEEE},
  keywords={Prototypes;Anomaly detection;Feature extraction;Self-supervised learning;Spatiotemporal phenomena;Behavioral sciences;Tensors;Anomaly detection;crowd management;urban transportation systems;contrastive learning},
  doi={10.1109/TITS.2024.3355143}
}
```

## Acknowledgement

We appreciate the following github repos a lot for their valuable code:

- https://github.com/dleyan/STGAN

- https://github.com/RElbers/info-nce-pytorch

- https://github.com/uchidalab/time_series_augmentation
