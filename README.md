# Distributed Convolutional Neural Networks for the Internet-of-Things
This repository provides the python code of the "Distributed Convolutional Neural Networks for the Internet-of-Things" paper.

## Citation and Contact

If you use our work, please also cite the paper:
```
@article{disabato2019distributed,
  title={Distributed Deep Convolutional Neural Networks for the Internet-of-Things},
  author={Disabato, Simone and Roveri, Manuel and Alippi, Cesare},
  journal={arXiv preprint arXiv:1908.01656},
  year={2019}
}
```

## Abstract
> >  Severe constraints on memory and computation characterizing the Internet-of-Things (IoT) units may prevent the execution of Deep Learning (DL)-based solutions, which typically demand large memory and high processing load. In order to support a real-time execution of the considered DL model at the IoT unit level, DL solutions must be designed having in mind constraints on memory and processing capability exposed by the chosen IoT technology. In this paper, we  introduce a design methodology aiming at allocating the execution of Convolutional Neural Networks (CNNs) on a distributed IoT application. Such a methodology is formalized as an optimization problem where the latency between the data-gathering phase and the subsequent decision-making one is minimized, within the given constraints on memory and processing load at the units level. 
> > The methodology supports multiple sources of data as well as multiple CNNs in execution on the same IoT system allowing the design of CNN-based applications demanding autonomy, low decision-latency, and high Quality-of-Service.


## Installation
This code is written in `Python 3.7` and requires the packages listed in `requirements.txt`, along with a valid license of the Gurobi optimizer (https://www.gurobi.com).

Clone the repository to your local machine in the desired directory:
```
git clone https://github.com/simdis/distributedCNNs
```

To run the code, a virtual environment is highly suggested, e.g. using `virtualenv` or `conda`:

### `virtualenv`
```
# pip install virtualenv
cd <path-to-distributedCNNs-directory>
virtualenv envname
source envname/bin/activate
pip install -r requirements.txt
```

### `conda`
```
cd <path-to-distributedCNNs-directory>
conda create --name envname
source activate envname
while read requirement; do conda install -n envname --yes $requirement; done < requirements.txt
```
