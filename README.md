# Piforest 🌳🌲🌳🌲

Implementation of the *anomaly detection in resource constrained environments with streaming data* by [jain et al. (2021)](https://ieeexplore.ieee.org/abstract/document/9410461).

>Jain, Prarthi, Seemandhar Jain, Osmar R. Zaïane, and Abhishek Srivastava. "Anomaly Detection in Resource Constrained Environments With Streaming Data." IEEE Transactions on Emerging Topics in Computational Intelligence (2021).

## About

The Preprocessed isolation forest (PiForest) algorithm is a method for detecting outliers in streaming data for resource constrained environment. PiForest offers a number of features that many competing anomaly detection algorithms lack. Specifically, PiForest:

- Is designed to handle streaming data.
- Performs well in resource constrained environment.
- Performs well on high-dimensional data.
- Reduces the influence of irrelevant dimensions.
- Features an anomaly-scoring algorithm with a clear underlying statistical meaning.

This repository provides an open-source implementation of the PiForest algorithm and its core data structures for the purposes of facilitating experimentation and enabling future extensions of the PiForest algorithm.

## Documentation

Read the docs [here 📖](https://www.piforest.ml/).

## Installation

Download the repo, and run the main file.
Currently, only Python 3 is supported.

### Dependencies

The following dependencies are *required* to install and use `PiForest`:

-numpy==1.19.2
-pandas==1.1.3
-scikit_multiflow==0.5.3
-scikit_learn==0.24.2


Listed version numbers have been tested and are known to work (this does not necessarily preclude older versions).


## Citing

If you have used this codebase in a publication and wish to cite it, please use the [`IEEE Transactions on Emerging Topics in Computational Intelligence`](https://ieeexplore.ieee.org/abstract/document/9410461).

> Jain, Prarthi, Seemandhar Jain, Osmar R. Zaïane, and Abhishek Srivastava. "Anomaly Detection in Resource Constrained Environments With Streaming Data." IEEE Transactions on Emerging Topics in Computational Intelligence (2021).

```bibtex
@article{jain2021anomaly,
  title={Anomaly Detection in Resource Constrained Environments With Streaming Data},
  author={Jain, Prarthi and Jain, Seemandhar and Za{\"\i}ane, Osmar R and Srivastava, Abhishek},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence},
  year={2021},
  publisher={IEEE}
}
```

# PiForest Working
The authors present the Preprocessed Isolation Forest PiForest approach for anomaly detection that works well in resource constrained environments and is also effective on streaming data.
Propsed Architecture
![alt text](https://raw.githubusercontent.com/jainsee24/PiForest/main/Approach/jain3%20(2)-1.jpg)

Cicular Queue
![alt text](https://raw.githubusercontent.com/jainsee24/PiForest/main/Approach/jain5%20(1)-1.jpg)

Arduino Real world Set-up

![alt text](https://raw.githubusercontent.com/jainsee24/PiForest/main/Approach/jain10-1.jpg)
