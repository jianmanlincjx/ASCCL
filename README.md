# [CVPR 2024 Highlight (11.9%)] LucidDreamer: Towards High-Fidelity Text-to-3D Generation via Interval Score Matching

<div style="text-align: justify;">

Speech-preserving facial expression manipulation (SPFEM) aims to modify facial emotions while meticulously maintaining the mouth animation associated with spoken content. Current works depend on inaccessible paired training samples for the person, where two aligned frames exhibit the same speech content yet differ in emotional expression, limiting the SPFEM applications in real-world scenarios. In this work, we discover that speakers who convey the same content with different emotions exhibit highly correlated local facial animations, providing valuable supervision for SPFEM. To capitalize on this insight, we propose a novel adaptive spatial coherent correlation learning (ASCCL) algorithm, which models the aforementioned correlation as an explicit metric and integrates the metric to supervise manipulating facial expression and meanwhile better preserving the facial animation of spoken contents. To this end, it first learns a spatial coherent correlation metric, ensuring the visual disparities of adjacent local regions of the image belonging to one emotion are similar to those of the corresponding counterpart of the image belonging to another emotion. Recognizing that visual disparities are not uniform across all regions, we have also crafted a disparity-aware adaptive strategy that prioritizes regions that present greater challenges. During SPFEM model training, we construct the adaptive spatial coherent correlation metric between corresponding local regions of the input and output images as addition loss to supervise the generation process. We conduct extensive experiments on variant datasets, and the results demonstrate the effectiveness of the proposed ASCCL algorithm.

</div>

---

## 1. Installation

Create a conda environment and install the requirements.

```bash
conda env create -f environment.yml
conda activate ASCCL
```
Follow the instructions in [DECA](https://github.com/yfeng95/DECA) (under the *Prepare data* section) to acquire the 3 files (`generic_model.pkl`, `deca_model.tar`, `FLAME_albedo_from_BFM.npz`) and place them under `./DECA/data`.

