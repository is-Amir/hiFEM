# hiFEM: Hyperelastic Inverse Finite Element Method

This repository contains the implementation of the Hyperelastic Inverse Finite Element Method (`hiFEM`), a novel computational tool for patient-specific virtual surgical planning (VSP) in soft tissue reconstruction. hiFEM is a biomechanics-informed VSP algorithm designed to optimize free flap designs for reconstructive surgeries. The core principle of hiFEM is to determine the optimal planar flap shape by minimizing the deformation energy within the tissue as it deforms to match the patient-specific 3D defect geometry. This physics-based energy minimization approach distinguishes hiFEM from purely geometric flattening methods; It generates surgical designs that require less tissue stretch during the 2D-to-3D transformation. 

## Applications

While initially developed for tongue reconstruction VSP after tumor removal, hiFEM's fundamental principle of minimizing tissue stretches makes it broadly applicable across various soft tissue reconstructive surgery domains. This includes, but is not limited to:

- **Tongue Reconstruction**: Designing precise free flaps to restore anatomical integrity and function after tumor resection and reconstruciton.
- **Breast Reconstruction**: Planning tailored flap geometries for breast reconstruction procedures.
- **Nasal Reconstruction**: Optimizing flap designs for complex nasal defects.
- **Complex Wound Closures**: Assisting in planning skin grafts and flap closures where minimizing tensions and scaring is crucial.

By integrating patient-specific anatomy and biomechanics, hiFEM provides a pathway to enhance precision and predictability in challenging reconstructive scenarios

## Installation

To set up the environment and run the `hiFEM` project, please follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/is-amir/hifem.git
    cd hifem
    ```

2.  **Create and Activate Conda Environment:**
    Open your terminal or Anaconda Prompt and run the following commands to create a new Conda environment. Feel free to change the env name (`hifem_env`)

    ```bash
    conda create -n hifem_env python=3.11
    conda activate hifem_env
    ```

3.  **Install Dependencies:**
    With the environment activated, install all the required packages using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

## Usage
  * **To use `hiFEM` as a function:**
    You can call `hiFEM` function with your own 3D mesh data. Below is a simplified example showing how to load a mesh, run the function, and save the output:

    ```python
    import pyvista as pv
    from pathlib import Path
    import os
    import src

    # Create an 'outputs' directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    # Load your 3D mesh (e.g., "inputs/flap3D.obj")
    mesh_3d = pv.read(Path("inputs/flap3D.obj")).triangulate()
    verts_3d = mesh_3d.points
    faces_3d = mesh_3d.faces.reshape(-1, 4)[:, 1:]

    # Run hiFEM
    # This will generate various visualization output files as well if verbose is True.
    mesh_2d, _ = src.hiFEM(
        verts3D=verts_3d,
        faces=faces_3d,
        verbose=True,
    )

    # Save the resulting 2D mesh
    mesh_2d.save("outputs/my_flattened_mesh.obj", binary=False)
    ```

  * **To regenerate the paper's figures run the `regenerate_paper_figures.ipynb` notebook:**
    This notebook runs the `hiFEM` function for several pre-defined cases (Case A, B, C, D) using different material models and initial guess methods to reproduce paper figures.

## Citing
If you find this repository and the hiFEM method useful for your research or projects, we kindly request that you cite our paper. Your citation helps acknowledge our work and supports the open-source community.

```bibtex
@article{isazadeh2025patient,
  title={Patient-specific virtual surgical planning for tongue reconstruction: Evaluating hyperelastic inverse FEM with four simulated tongue cancer cases},
  author={Isazadeh, Amir R and Zenke, Julianna and Westover, Lindsey and Seikaly, Hadi and Aalto, Daniel},
  journal={under review},
  year={2025}
}
```