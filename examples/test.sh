#!/bin/bash
# rm slurm-*.out
ml load qmio/hpc gcc/12.3.0 hpcx-ompi flexiblas/3.3.0 boost cmake/3.27.6 gcccore/12.3.0 nlohmann_json/3.11.3 ninja/1.9.0 pybind11/2.13.6-python-3.11.9 qiskit/1.2.4-python-3.11.9 rust qmio-run qmio-tools
source ../.env3/bin/activate
export ZMQ_SERVER=tcp://10.255.3.70:5556
# python examples/teleport_polypus.py
# sleep 2
python examples/qft.py