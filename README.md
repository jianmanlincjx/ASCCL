# [CVPR 2024 Highlight (11.9%)] Learning Adaptive Spatial Coherent Correlations for Speech-Preserving Facial Expression Manipulation
## Requirements
- Download the MEAD dataset from ([here](https://wywu.github.io/projects/MEAD/MEAD.html)).
- Download the pre-trained weights ([here](https://drive.google.com/file/d/1W_qa9xxXTCXo_44PX_oRDLlJQ3F8uXJk/view?usp=sharing)) (" backbone.pth ") 

## Preprocessing
The obtained MEAD dataset is first preprocessed with 'align_face.py':
```bash
python align_face.py
```
Paired image frames of the same speaker saying the same sentence with different emotions are recorded in aligned_path36.json

## Training
To train the model, run './trainer/train_asccl.py' with the preprocessed dataset path configured:
```bash
python ./trainer/train_asccl.py
```

## Integration into SPFEM models（take NED as example）
NED：([code](https://github.com/foivospar/NED)).

Integrate ASCCL into NED's training process: 

1. First follow NED's data preprocessing process to obtain training data and model parameters
2.
3. 2. Replace the train.py file in NED's manipulator folder with the train_ned.py file
