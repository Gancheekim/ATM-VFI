# ATM-VFI: Exploiting Attention-to-Motion via Transformer for Versatile Video Frame Interpolation

In this repository, we present a versatile VFI work, utilizing the Attention-to-Motion (ATM) module to intuitively formulate motion estimation.

- Paper: (Under review)
- Video demo: [Youtube](https://www.youtube.com/watch?v=bSdBEfe9haM)

## Architecture Overview
<img src="./asset/model-overview-ver3.png" alt="drawing" height="260"/>

## Attention-to-Motion
<img src="./asset/atm_working-example.png" alt="drawing" height="265"/>&nbsp;&nbsp;<img src="./asset/ATMFormer_ver6.png" alt="drawing" height="265"/>

## Dependencies
We provide the dependencies in `requirements.txt`.

## Demo
For 2x interpolation, run the command below:
> use `--global_off` flag to disable the global motion estimation.
- input: 2 frames
    ```
    python3 demo_2x.py --model_type <select base or lite> --ckpt <path to model checkpoint> --frame0 <path to frame 0> --frame1 <path to frame 1> --out <path to output frame>
    ```
- input: mp4 video
    ```
    python3 demo_2x.py --model_type <select base or lite> --ckpt <path to model checkpoint> --video <path to .mp4 file>
    ```
    > use `--combine_video` flag to combine the original input video and processed video.
    

### Example: 2x interpolation comparison (24 fps v.s. 48 fps)
<video width="640" height="720" controls>
  <source src="./asset/output_interpolation._combine.mp4" type="video/mp4">
</video>

## Pretrained checkpoints
We will release the checkpoints after the final paper decision.
|Version|Link|Param (M)|
|-------|----|---------|
|Base   |TBA |51.56|
|Lite   |TBA |11.98|
|Pct    |TBA |51.56|

## Evalution
We evaluate our method using the `benchmark` scripts provided by [RIFE](https://github.com/hzwer/ECCV2022-RIFE/tree/main), [EMA-VFI](https://github.com/MCG-NJU/EMA-VFI/tree/main) and [AMT](https://github.com/MCG-NKU/AMT?tab=readme-ov-file) for consistency. 
- Vimeo90K
    ```
    cd benchmark
    python3 test_vimeo90k.py --path <path to Vimeo90K dataset folder> --ckpt <path to model checkpoint>
    ```
- UCF101
    ```
    cd benchmark
    python3 test_ucf101.py --path <path to UCF101 dataset folder> --ckpt <path to model checkpoint>
    ```
- SNU-FILM
    ```
    cd benchmark
    python3 test_snufilm.py --path <path to SNU-FILM dataset txt> --img_data_path <path to SNU-FILM dataset image folder> --ckpt <path to model checkpoint>
    ```
- Xiph
    ```
    cd benchmark
    python3 test_xiph.py --root <path to Xiph dataset folder> --ckpt <path to model checkpoint>
    ```

## Training/Fine-tuning
The first 2 phases of the training procedure (stated in our paper) utilize `train.py` and `trainer.py`, while the last 2 phases utilize `finetune.py` and `finetune_trainer.py`.
- Phase 1: run `train.py` and set the argument `dataset` as `vimeo90k`, the other training hyperparameters can be set as you wish (batch size, learning rate, no. of epoch, etc.). Reminder: please make sure to uncomment `model.global_motion = False`.
- Phase 2: run `train.py`, set the argument `dataset` as `X4k`, and remember to set the variable `isLoadCheckpoint` to `True` and change `param` to the checkpoint of Phase 1. Reminder: please make sure to uncomment `model.global_motion = True` and `model.__freeze_local_motion__()`.
- Phase 3: run `finetune.py`, change `param` to the checkpoint of Phase 2. For more tweaking, please trace the source code.
- Phase 4: run `finetune.py`, change `param` to the checkpoint of Phase 3.

## Citation
TBA

## Acknowledgement
