#  Experiments with Synthetic Data for Human-in-the-Loop Large-Scale Predictive Maintenance of Workstations

This repository contains toy experiments with synthetic data for the publication
* Alexander Nikitin and Samuel Kaski (2022). **Human-in-the-Loop Large-Scale Predictive Maintenance of Workstations**. 

For production implementation, check Sections "5. Industrial Implementation" and "6.2 Online Experiments" of the paper.

<p align="center">
  <img src="data/teaser.png" width="500" height="300"/>
</p>


## Use
The repo uses [git-lfs](https://git-lfs.github.com/) to store datasets. To fetch the data use:
```bash
git lfs fetch
```

The code was tested with `python>=3.6`.

To install the required packages, run:
```bash
pip install -r requirements.txt
pip install -e .
```

## Structure
The repository contains and implementation of the approach for predictive maintenance of the worksations. The structure is as follows:
* `dre_pdm` contains utilities for decision rule elicitation modeling and training of the models,
* `experiments/simulator.ipynb` contains an implementation of the synthetic data simulator,
* `experiments/analysis.ipynb` contains the experiments with synthetic data from the article, and visualizations,
* `data` contains a generated dataset.

## Experiments.

### Simulator Experiments:
Open with jupyter-notebook:
```bash
./experiments/analysis.ipynb
```

## Citation
If you found the publication useful for your research, please cite the paper as follows:
```bibtex
@inproceedings{nikitin2022pdm,
  title={Alexander Nikitin and Samuel Kaski},
  author={Human-in-the-Loop Large-Scale Predictive Maintenance of Workstations},
  year={2022}
}
```

## License
This software is provided under the [Apache License 2.0](LICENSE).