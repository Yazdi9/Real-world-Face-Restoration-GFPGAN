
### GFP-GAN: Blind Face Restoration with Generative Facial Prior

### Output
<p align="center">
  
  <img src="output/download (1).png">
  <img src="output/download (5).png">
  <img src="output/download (9).png">
  <img src="output/download (6).png">
  <img src="output/download (7).png">
  <img src="output/download (8).png">
</p>

---

### Installation
1. Clone repo

    ```bash
    git clone https://github.com/saba99/Real-world-Face-Restoration-GFPGAN.git
    cd Real-world-Face-Restoration-GFPGAN
    ```

1. Install dependent packages

    ```bash
    # Install basicsr - https://github.com/xinntao/BasicSR
    # We use BasicSR for both training and inference
    pip install basicsr

    # Install facexlib - https://github.com/xinntao/facexlib
    # We use face detection and face restoration helper in the facexlib package
    pip install facexlib

    pip install -r requirements.txt
    python setup.py develop

    # If you want to enhance the background (non-face) regions with Real-ESRGAN,
    # you also need to install the realesrgan package
    pip install realesrgan
    ```

## Quick Inference

We take the v1.3 version for an example. More models can be found [here](#european_castle-model-zoo).

Download pre-trained models: [GFPGANv1.3.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth)

```bash
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models
```

**Inference!**

```bash
python inference_gfpgan.py -i inputs/whole_imgs -o results -v 1.3 -s 2
```

##  Model 

| Version | Model Name  | Description |
| :---: | :---:        |     :---:      |
| V1.3 | [GFPGANv1.3.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth) | Based on V1.2; **more natural** restoration results; better results on very low-quality / high-quality inputs. |
| V1.2 | [GFPGANCleanv1-NoCE-C2.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth) | No colorization; no CUDA extensions are required. Trained with more data with pre-processing. |
| V1 | [GFPGANv1.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth) | The paper model, with colorization. |

## Training

We provide the training codes for GFPGAN (used in our paper). <br>
You could improve it according to your own needs.


**Procedures**

1. Dataset preparation: [FFHQ](https://github.com/NVlabs/ffhq-dataset)

1. Download pre-trained models and other data. Put them in the `experiments/pretrained_models` folder.
    1. [Pre-trained StyleGAN2 model: StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth)
    1. [Component locations of FFHQ: FFHQ_eye_mouth_landmarks_512.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/FFHQ_eye_mouth_landmarks_512.pth)
    1. [A simple ArcFace model: arcface_resnet18.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/arcface_resnet18.pth)

1. Modify the configuration file `options/train_gfpgan_v1.yml` accordingly.

1. Training

> python -m torch.distributed.launch --nproc_per_node=4 --master_port=22021 gfpgan/train.py -opt options/train_gfpgan_v1.yml --launcher pytorch

