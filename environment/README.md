
# Table of Contents

1.  [Adapt code to M2 mac](#org055ac73)
    1.  [Adapt environtment](#org2ea0233)
        1.  [Solving the `Intel MKL FATAL ERROR`](#org97120c1)
    2.  [Avoid GPU use+](#orgccdcd3a)


<a id="org055ac73"></a>

# Adapt code to M2 mac


<a id="org2ea0233"></a>

## Adapt environtment

Write the following in the `environment/vtoonify_env_mac.yaml` file

    name: vtoonify_env
    channels:
      - pytorch
      - conda-forge
      - defaults
    dependencies:
      - _libgcc_mutex
      - ca-certificates
      - certifi
      - faiss
      - libedit
      - libfaiss
      - libfaiss-avx2
      - libffi
      - matplotlib-base
      - mkl < 2022
      - pillow
      - pip
      - python
      - python-lmdb
      - pytorch
      - setuptools
      - scikit-image
      - torchaudio
      - torchvision
      - pip:
        - cmake
        - dlib
        - matplotlib
        - ninja
        - numpy
        - opencv-python
        - scipy
        - tqdm
        - wget

    CONDA_SUBDIR=osx-64 conda env create -f ./environment/vtoonify_env_mac.yaml
    conda activate vtoonify_env


<a id="org97120c1"></a>

### Solving the `Intel MKL FATAL ERROR`

Add the following to the `environment/vtoonify_env_mac.yaml` file

    - mkl < 2022

    CONDA_SUBDIR=osx-64 conda env update -f ./environment/vtoonify_env_mac.yaml

Solution shown [here](https://stackoverflow.com/questions/70830755/intel-mkl-fatal-error-this-system-does-not-meet-the-minimum-requirements-for-us).


<a id="orgccdcd3a"></a>

## Avoid GPU use+

Make the changes described [here](https://github.com/Malnascudes/VToonify/tree/main/model/stylegan/op_cpu#readme). Change `model.stylegan.op` to `model.stylegan.op_cpu`

-   <https://github.com/williamyang1991/VToonify/blob/01b383efc00007f9b069585db41a7d31a77a8806/util.py#L14>
-   <https://github.com/williamyang1991/VToonify/blob/01b383efc00007f9b069585db41a7d31a77a8806/model/simple_augment.py#L12>
-   <https://github.com/williamyang1991/VToonify/blob/01b383efc00007f9b069585db41a7d31a77a8806/model/stylegan/model.py#L11>

