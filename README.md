# ATM-VFI: Exploiting Attention-to-Motion via Transformer for Versatile Video Frame Interpolation

In this repository, we present a versatile VFI work, utilizing the Attention-to-Motion (ATM) module to intuitively formulate motion estimation.

- Paper: (Under review)
- Video demo: [Youtube](https://www.youtube.com/watch?v=bSdBEfe9haM)

## Architecture Overview
<img src="./asset/model-overview-ver3.png" alt="drawing" height="300"/>
<br />

## Attention-to-Motion
<img src="./asset/atm_working-example.png" alt="drawing" height="300"/>&nbsp;&nbsp;&nbsp;&nbsp;
<img src="./asset/ATMFormer_ver6.png" alt="drawing" height="300"/>

## Dependencies
We provide the dependencies in `requirements.txt`.

## Demo
```
python3 demo_2x.py --model_type <select base or lite> --ckpt <path to model checkpoint> --frame0 <path to frame 0> --frame1 <path to frame 1> --out <path to output frame>
```

## Pretrained checkpoints
TBA, we will release the checkpoints after the final paper decision.

## Evalution
We evaluate our method using the `benchmark` scripts provided by [RIFE]() and [EMA-VFI] for consistency. 

## Citation
TBA
