celeb=$1
mode=$2
# python renderer/train.py --celeb /data2/JM/code/NED-main/train_render_dataset_train/W029 --checkpoints_dir /data2/JM/code/NED-main/renderer_checkpoints_me/W029 --load_pretrain /data2/JM/code/NED-main/renderer_checkpoints_author/Meta-renderer/checkpoints_meta-renderer --which_epoch 15
if [ "$mode" = "train" ]
then
    echo "--------------- Preprocessing mode: train ---------------"
    echo
    # python preprocessing/detect.py --celeb $celeb --split
    # python preprocessing/eye_landmarks.py --celeb $celeb --mouth --align
    # python preprocessing/segment_face.py --celeb $celeb
    # python preprocessing/reconstruct.py --celeb $celeb \
    #                                     --save_shapes \
    #                                     --save_nmfcs
    # python preprocessing/align.py --celeb $celeb --faces_and_masks \
    #                                              --shapes \
    #                                              --nmfcs \
    #                                              --landmarks

elif [ "$mode" = "test" ]
then
  echo "--------------- Preprocessing mode: test ---------------"
  echo
  # python preprocessing/detect.py --celeb $celeb --save_videos_info --save_full_frames
  # python preprocessing/eye_landmarks.py --celeb $celeb --align
  # python preprocessing/segment_face.py --celeb $celeb
  python preprocessing/reconstruct.py --celeb $celeb \
                                      --save_shapes \
                                      --save_nmfcs
  python preprocessing/align.py --celeb $celeb --faces_and_masks \
                                               --shapes \
                                               --nmfcs \
                                               --landmarks

elif [ "$mode" = "reference" ]
then
  echo "--------------- Preprocessing mode: reference ---------------"
  echo
  python preprocessing/detect.py --celeb $celeb --save_full_frames
  python preprocessing/reconstruct.py --celeb $celeb

else
    echo "Invalid mode given"
fi
