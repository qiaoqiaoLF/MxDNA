# MxDNA

This is the repository for *Model Decides How to Tokenize: Adaptive DNA Sequence Tokenization with MxDNA*.

## The implementation of the Learnt Tokenization Module in MxDNA

We have extracted the core implementation of the Learnt Tokenization Module and provide it here for anyone who is interested. To access the full implementation, please check out the full-model branch.

### File Structure

The implementation of the Learnt Tokenization Module in MxDNA is located in the `mxdna` directory. The `mxdna` directory contains the following files:

- `README.md`: This file provides an overview of the implementation of the Learnt Tokenization Module in MxDNA.
- `mxdna.py`: This file contains the implementation of the Learnt Tokenization Module in MxDNA.
- `BasicUnitNMS.cpp`: This file contains the implementation of the basic unit non-maximum suppression (NMS) algorithm used in MxDNA.
- `CMakeLists.txt`: This file contains the CMake configuration for building the MxDNA project.

You need to further clone the pybind11 repository to compile the `BasicUnitNMS.cpp` file into a shared object file for use in the `mxdna.py` file.

- `pybind11`: This directory contains the pybind11 library used for Python bindings in MxDNA. You need to clone the pybind11 repository to use MxDNA.
- `BasicUnitNMS.cpython-{$PYTHONVERSION}-{$SYSTEMARCHITECTURE}-linux-gnu.so`: This file contains the compiled shared object file for the basic unit NMS algorithm. You need to compile this file using the provided CMake configuration.

### Learnt Tokenization Module

The core of the Learnt Tokenization Module in MxDNA is the `MxDNALearntTokenizationLayer` class defined in `mxdna.py`. The Non-maximum Suppression (NMS) algorithm is implemented in the `BasicUnitNMS.cpp` file. It is compiled into a python packaging using pybind11 and used in the `MxDNALearntTokenizationLayer` class. The sparse Mixture of Convolution Experts is the `MxDNAConvMoeBlock` class defined in `mxdna.py`. The deformable convolution is the `MxDNADeforambleConvBlock` class defined in `mxdna.py`. The comments in the code provide detailed explanations of the implementation.

### Glossary of Terms


| **Term**                               | **Description**                                                 | **Variable in Code**                                             |
|----------------------------------------|-----------------------------------------------------------------| -----------------------------------------------------------------|
| $l$                                 | Number of nucleotides                                            | `seq_len` before tokenization |
| $d$                                 | Dimension of hidden states                                       | `hidden_dim` |
| $n$                                 | Number of experts                                                | `num_experts` |
| $k$                                 | Number of basic units                                            | `seq_len` after tokenization |
| $f$                                 | Kernel size of deformable convolution                            | `deforamble_conv_kernel_size` |
| $i$                                 | Indices of nucleotides or tokens                                 |  not used |
| $j$                                 | Indices of experts                                               | `expert_idx` |
| $\mathbf{X} \in \mathbb{R}^{l \times d}$ | Input nucleotide sequence                                        |  `hidden_states` before tokenization |
| $\mathbf{S} \in \mathbb{R}^{l \times n}$ | Confidence scores of basic units existence                       | `router_logits` |
| $\mathbf{L} \in \mathbb{N}^{n}$         | Kernel sizes of convolution experts                              | `expert_kernel_sizes` |
| $\mathbf{M} \in \mathbb{N}^{l}$         | Mask of basic units existence                                    | `basic_unit_mask_center` |
| $\mathbf{E_j} \in \mathbb{R}^{L_j \times d} \rightarrow \mathbb{R}^d$| Convolution experts                      | `MxDNAConvMoeBlock.experts` |
| $\mathbf{U} \in \mathbb{R}^{k \times d}$ | Basic units                                                      | `hidden_states` after sparse mixture of convolution experts |
| $\Delta \mathbf{P} \in \mathbb{R}^{k \times f}$ | Offsets of deformable convolution                         | `offset` |
| $\Delta \mathbf{M} \in \mathbb{R}^{k \times f}$ | Modulation factors of deformable convolution             | `modulator` |
| $\mathbf{T} (\mathbf{Y}) \in \mathbb{R}^{k \times d}$ | Final tokens                                                     | `hidden_states` after deformable convolution |

