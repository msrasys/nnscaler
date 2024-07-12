==============================================================
nnScaler: A Parallelization System for DNN Model Training
==============================================================

Introduction
------------
**nnScaler** is a parallelization system for deep neural network (DNN) model training.


nnScaler automatically parallelizes DNN models across multiple devices, enabling users to focus on model design. nnScaler supports new parallelisms that outperform existing parallel execution approaches. nnScaler supports extending DNN modules with new structures or execution patterns, enabling users to parallelize their own new DNN models. nnScaler can support paralleling new DNN models by providing user-defined functions for the new operators unrecognized by the nnScaler.

Features
--------
- **Automatic Parallelization**: nnScaler automatically parallelizes DNN models across multiple devices, enabling users to focus on model design.
- **High Performance**: nnScaler supports new parallelisms that outperform existing parallel execution approaches.
- **Extensibility**: nnScaler supports extending DNN modules with new structures or execution patterns, enabling users to parallelize their own new DNN models.
- **Compatibility**: nnScaler can support paralleling new DNN models by providing user-defined functions for the new operators unrecognized by the nnScaler.

Overview
--------

Below is an overview of the nnScaler system. The nnScaler system consists of three main components: the parallelization compiler, the planner, and the interface. The parallelization compiler takes a DNN model as input, converts into intermediate representation (Graph IR) and generates execution for multiple devices. The parallelization planner will provide efficient strategies during parallelization. The nnScaler interface provides a set of parallelization APIs to support different trainers through certain adapters, as well as extending the nnScaler system.

.. figure:: images/overview.png
    :alt: overview
    :figwidth: 80%
    :align: center

    **nnScaler Overview**

Outline
--------
- **Quick Start**: Learn how to install and use nnScaler.
    - **Installation**: Install nnScaler on your machine.
    - **Get Started**: Started from a simple example.
- **User Guide**: Learn how to use nnScaler to parallelize a model.
    - **Example**: Parallelize NanoGPT through PyTorch Lightning interface.
- **Developer Guide**: Find detailed information about nnScaler.
    - **Extending nnScaler**: Learn how to extend nnScaler.
- **Frequently Asked Questions**: Find answers to common questions about nnScaler.


Reference
---------
Please cite nnScaler in your publications if it helps your research: ::

    @inproceedings {nnscaler-osdi24,
    author = {Zhiqi Lin and Youshan Miao and Quanlu Zhang and Fan Yang and Yi Zhu and Cheng Li and Saeed Maleki and Xu Cao and Ning Shang and Yilei Yang and Weijiang Xu and Mao Yang and Lintao Zhang and Lidong Zhou},
    title = {nnScaler: Constraint-Guided Parallelization Plan Generation for Deep Learning Training},
    booktitle = {18th {USENIX} Symposium on Operating Systems Design and Implementation ({OSDI} 24)},
    year = {2024},
    publisher = {{USENIX} Association},
    }

Contributing
------------
This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the `Microsoft Open Source Code of Conduct <https://opensource.microsoft.com/codeofconduct/>`_. For more information, see the `Code of Conduct FAQ <https://opensource.microsoft.com/codeofconduct/faq/>`_ or contact `opencode@microsoft.com <mailto:opencode@microsoft.com>`_ with any additional questions or comments.

Trademarks
----------
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow `Microsoft's Trademark & Brand Guidelines <https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general>`_. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos is subject to those third-party's policies.

Contact
-------
You may find our public repo from `https://github.com/microsoft/nnscaler`_ or microsoft internal repo `https://aka.ms/ms-nnscaler`_.

.. _`https://github.com/microsoft/nnscaler`: https://github.com/microsoft/nnscaler

.. _`https://aka.ms/ms-nnscaler`: https://aka.ms/ms-nnscaler

For any questions or inquiries, please contact us at nnscaler@service.microsoft.com.

