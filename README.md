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
```
python3 demo_2x.py --model_type <select base or lite> --ckpt <path to model checkpoint> --frame0 <path to frame 0> --frame1 <path to frame 1> --out <path to output frame>
```

## Pretrained checkpoints
We will release the checkpoints after the final paper decision.
|Version|Link|Param (M)|
|-------|----|---------|
|Base   |TBA |51.56|
|Lite   |TBA |11.98|
|Pct    |TBA |51.56|

## Evalution
We evaluate our method using the `benchmark` scripts provided by [RIFE](https://github.com/hzwer/ECCV2022-RIFE/tree/main) and [EMA-VFI](https://github.com/MCG-NJU/EMA-VFI/tree/main) for consistency. 
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

## Citation
TBA
