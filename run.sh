CUDA_VISIBLE_DEVICES=2 python manipulator/train.py \
    --train_root /data2/JM/code/NED-main/MEAD_data \
    --selected_actors M003 M009 W029 \
    --selected_actors_val M023 \
    --checkpoints_dir manipulator_checkpoints/manipulator_checkpoints_pretrained_affwild2 --finetune

CUDA_VISIBLE_DEVICES=3 python renderer/train.py --celeb RAVDESS/Train/Actor_01 --checkpoints_dir RAVDESS_checkpoint/Actor_01 --load_pretrain renderer_checkpoints_author/Meta-renderer/checkpoints_meta-renderer --which_epoch 15

# CUDA_VISIBLE_DEVICES=1 python manipulator/test.py --celeb test/FrancesMcDormand_t --checkpoints_dir manipulator_checkpoints/manipulator_checkpoints_pretrained_affwild2_author \
#                            --ref_dirs reference/Pacino_clip/DECA --exp_name Pacino_clip_3 --which_epoch 3

# CUDA_VISIBLE_DEVICES=1 python manipulator/test.py --celeb test/FrancesMcDormand_t --checkpoints_dir manipulator_checkpoints/manipulator_checkpoints_pretrained_affwild2 \
#                            --ref_dirs reference/Pacino_clip/DECA --exp_name Pacino_clip_4 --which_epoch 4

# CUDA_VISIBLE_DEVICES=1 python manipulator/test.py --celeb test/FrancesMcDormand_t --checkpoints_dir manipulator_checkpoints/manipulator_checkpoints_pretrained_affwild2 \
#                            --ref_dirs reference/Pacino_clip/DECA --exp_name Pacino_clip_5 --which_epoch 5

# CUDA_VISIBLE_DEVICES=1 python manipulator/test.py --celeb test/FrancesMcDormand_t --checkpoints_dir manipulator_checkpoints/manipulator_checkpoints_pretrained_affwild2 \
#                            --ref_dirs reference/Pacino_clip/DECA --exp_name Pacino_clip_6 --which_epoch 6

# CUDA_VISIBLE_DEVICES=1 python manipulator/test.py --celeb test/FrancesMcDormand_t --checkpoints_dir manipulator_checkpoints/manipulator_checkpoints_pretrained_affwild2 \
#                            --ref_dirs reference/Pacino_clip/DECA --exp_name Pacino_clip_7 --which_epoch 7

# CUDA_VISIBLE_DEVICES=1 python manipulator/test.py --celeb test/FrancesMcDormand_t --checkpoints_dir manipulator_checkpoints/manipulator_checkpoints_pretrained_affwild2 \
#                            --ref_dirs reference/Pacino_clip/DECA --exp_name Pacino_clip_8 --which_epoch 8

# CUDA_VISIBLE_DEVICES=1 python manipulator/test.py --celeb test/FrancesMcDormand_t --checkpoints_dir manipulator_checkpoints/manipulator_checkpoints_pretrained_affwild2 \
#                            --ref_dirs reference/Pacino_clip/DECA --exp_name Pacino_clip_9 --which_epoch 9

# CUDA_VISIBLE_DEVICES=1 python manipulator/test.py --celeb test/FrancesMcDormand_t --checkpoints_dir manipulator_checkpoints/manipulator_checkpoints_pretrained_affwild2 \
#                            --ref_dirs reference/Pacino_clip/DECA --exp_name Pacino_clip_10 --which_epoch 10


# CUDA_VISIBLE_DEVICES=2 sh ./postprocess.sh test/FrancesMcDormand_t Pacino_clip_2 renderer_checkpoints_author/McDormand
# CUDA_VISIBLE_DEVICES=1 sh ./postprocess.sh test/FrancesMcDormand_t Pacino_clip_3 renderer_checkpoints_author/McDormand
# CUDA_VISIBLE_DEVICES=1 sh ./postprocess.sh test/FrancesMcDormand_t Pacino_clip_4 renderer_checkpoints_author/McDormand
# CUDA_VISIBLE_DEVICES=1 sh ./postprocess.sh test/FrancesMcDormand_t Pacino_clip_5 renderer_checkpoints_author/McDormand
# CUDA_VISIBLE_DEVICES=1 sh ./postprocess.sh test/FrancesMcDormand_t Pacino_clip_6 renderer_checkpoints_author/McDormand
# CUDA_VISIBLE_DEVICES=1 sh ./postprocess.sh test/FrancesMcDormand_t Pacino_clip_7 renderer_checkpoints_author/McDormand
# CUDA_VISIBLE_DEVICES=1 sh ./postprocess.sh test/FrancesMcDormand_t Pacino_clip_8 renderer_checkpoints_author/McDormand
# CUDA_VISIBLE_DEVICES=1 sh ./postprocess.sh test/FrancesMcDormand_t Pacino_clip_9 renderer_checkpoints_author/McDormand
# CUDA_VISIBLE_DEVICES=1 sh ./postprocess.sh test/FrancesMcDormand_t Pacino_clip_10 renderer_checkpoints_author/McDormand


# CUDA_VISIBLE_DEVICES=1 python postprocessing/images2video.py --imgs_path test/FrancesMcDormand_t/Pacino_clip_2/full_frames --out_path result_cp/train_mapi_2.mp4 --audio test/FrancesMcDormand_t/videos/FrancesMcDormand_t.mp4
# CUDA_VISIBLE_DEVICES=1 python postprocessing/images2video.py --imgs_path test/FrancesMcDormand_t/Pacino_clip_3/full_frames --out_path result_cp/train_mapi_3.mp4 --audio test/FrancesMcDormand_t/videos/FrancesMcDormand_t.mp4
# CUDA_VISIBLE_DEVICES=1 python postprocessing/images2video.py --imgs_path test/FrancesMcDormand_t/Pacino_clip_4/full_frames --out_path result_cp/train_mapi_4.mp4 --audio test/FrancesMcDormand_t/videos/FrancesMcDormand_t.mp4
# CUDA_VISIBLE_DEVICES=1 python postprocessing/images2video.py --imgs_path test/FrancesMcDormand_t/Pacino_clip_5/full_frames --out_path result_cp/train_mapi_5.mp4 --audio test/FrancesMcDormand_t/videos/FrancesMcDormand_t.mp4
# CUDA_VISIBLE_DEVICES=1 python postprocessing/images2video.py --imgs_path test/FrancesMcDormand_t/Pacino_clip_6/full_frames --out_path result_cp/train_mapi_6.mp4 --audio test/FrancesMcDormand_t/videos/FrancesMcDormand_t.mp4
# CUDA_VISIBLE_DEVICES=1 python postprocessing/images2video.py --imgs_path test/FrancesMcDormand_t/Pacino_clip_7/full_frames --out_path result_cp/train_mapi_7.mp4 --audio test/FrancesMcDormand_t/videos/FrancesMcDormand_t.mp4
# CUDA_VISIBLE_DEVICES=1 python postprocessing/images2video.py --imgs_path test/FrancesMcDormand_t/Pacino_clip_8/full_frames --out_path result_cp/train_mapi_8.mp4 --audio test/FrancesMcDormand_t/videos/FrancesMcDormand_t.mp4
# CUDA_VISIBLE_DEVICES=1 python postprocessing/images2video.py --imgs_path test/FrancesMcDormand_t/Pacino_clip_9/full_frames --out_path result_cp/train_mapi_9.mp4 --audio test/FrancesMcDormand_t/videos/FrancesMcDormand_t.mp4
# CUDA_VISIBLE_DEVICES=1 python postprocessing/images2video.py --imgs_path test/FrancesMcDormand_t/Pacino_clip_10/full_frames --out_path result_cp/train_mapi_10.mp4 --audio test/FrancesMcDormand_t/videos/FrancesMcDormand_t.mp4


ssh -p 31455 root@connect.westa.seetacloud.com
hWZqqIpffL