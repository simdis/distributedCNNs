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

## How to use
To have an idea of how the proposed methodology works, the example folder is the best choice.
Within it, there are four jupyter notebooks covering four possible applications of the methodology itself.

If desired, the python script `execute_notebook.py` is able to run many times the proposed methodology. For each experiment, a jupyter notebook is created, along with all the output CSVs. For instance, with the command
```
python execute_notebook.py --num_exps 50 --output_dir <path-to-output-folder>
```
the methology will be run 50 times in the default configuration, i.e., a single AlexNet CNN to be placed in an IoT system comprising 50 units (45% of BeagleBone AI, 45% of OrangePi Zero, and 10% of Raspberry Pi 3B+).
The parameters of the `execute_notebook.py` allow to define several custom configurations. However, the current version of the script allows to define IoT systems comprising only three types of IoT units, i.e., the BeagleBone AI, the OrangePi Zero, and the Raspberry Pi 3B+ (if you set the probability of one of those IoT units to zero, you can define an IoT system without that IoT unit).
