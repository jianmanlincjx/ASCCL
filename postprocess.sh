celeb=$1
exp_name=$2
checkpoints_dir=$3

# render
python renderer/create_inputs.py --celeb $celeb --exp_name $exp_name --save_shapes
python renderer/test.py --celeb $celeb --exp_name $exp_name --checkpoints_dir $checkpoints_dir --which_epoch 10
# python postprocessing/unalign.py --celeb $celeb --exp_name $exp_name
# python postprocessing/blend.py --celeb $celeb --exp_name $exp_name --save_images




## manipulator
# ./preprocess.sh test_examples/Pacino/ test
# ./preprocess.sh reference_examples/Nicholson_clip/ reference
# ./preprocess.sh reference_examples/Pacino_clip/ reference
# ./preprocess.sh reference_examples/DeNiro_clip/ reference