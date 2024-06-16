# [CVPR 2024 Highlight (11.9%)] Learning Adaptive Spatial Coherent Correlations for Speech-Preserving Facial Expression Manipulation
## Requirements
- Download the MEAD dataset from ([here](https://wywu.github.io/projects/MEAD/MEAD.html)).
- Download the pre-trained weights ([here](https://drive.google.com/file/d/1W_qa9xxXTCXo_44PX_oRDLlJQ3F8uXJk/view?usp=sharing)) (" backbone.pth ") 

## Preprocessing
The obtained MEAD dataset is first preprocessed with 'align_face.py':
```bash
python align_face.py
```
## Pair Data
Paired image frames of the same speaker saying the same passage with different emotions are recorded in aligned_path36.json

## Training

To train the model, run './trainer/train_asccl.py' with the preprocessed dataset path configured:

```bash
python ./trainer/train_asccl.py
```
