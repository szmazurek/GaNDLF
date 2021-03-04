# GaNDLF

A **G**ener**a**lly **N**uanced **D**eep **L**earning **F**ramework for segmentation, regression and classification.

## Why use this?

- Supports multiple
  - Deep Learning model architectures
  - Data dimensions (2D/3D)
  - Channels/images/sequences 
  - Prediction classes
  - Domain modalities (i.e., Radiology Scans and Digitized Histopathology Tissue Sections)
- Robust data augmentation, courtesy of [TorchIO](https://github.com/fepegar/torchio/)  
- Built-in nested cross validation (and related combined statistics), with support for parallel HPC-based computing
- Handles imbalanced classes (e.g., very small tumor in large organ)
- Multi-GPU (on the same machine - distributed) training
- Leverages robust open source software
- No need to write any code to generate robust models
- [Automatic mixed precision](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/) support

## Citation

Please cite the following article for GaNDLF:

```
@misc{gandlf2021,
      title={GaNDLF: A Generally Nuanced Deep Learning Framework for Scalable End-to-End Clinical Workflows in Medical Imaging}, 
      author={Sarthak Pati and Siddhesh P. Thakur and Megh Bhalerao and Ujjwal Baid and Caleb Grenko and Brandon Edwards and Micah Sheller and Jose Agraz and Bhakti Baheti and Vishnu Bashyam and Parth Sharma and Babak Haghighi and Aimilia Gastounioti and Mark Bergman and Bjoern Menze and Despina Kontos and Christos Davatzikos and Spyridon Bakas},
      year={2021},
      eprint={2103.01006},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Documentation

Start at https://cbica.github.io/GaNDLF/
