
# ChildDiffusion: Controlled Synthesis of Identity-Preserving Child Faces

**ChildDiffusion** is a multi-stage generative pipeline for synthesizing diverse, high-fidelity child facial images using a combination of diffusion models, LoRA fine-tuning, model merging, and ControlNet-based conditioning. This project addresses the need for ethically sourced, demographically balanced, and controllably augmented child face datasets.

---

## 🚀 Key Features

- Text-guided diffusion synthesis with concept-specific fine-tuning
- Separate training pipelines for BoyDiffusion and GirlDiffusion models
- Weighted model merging for unified synthesis
- ControlNet-based structural conditioning (pose, expression, accessories)
- Evaluation with CLIP Score, BRISQUE, KID, identity similarity, and t-SNE
- Open-source datasets with variations in ethnicity, expression, and pose

---

## 🧠 Model Architecture

The pipeline includes:
1. Seed generation using ChildGAN
2. Concept-wise LoRA fine-tuning (Boy/Girl models)
3. Prompt-guided synthesis using Stable Diffusion
4. ControlNet conditioning for spatial guidance
5. Model merging to create the ChildDiffusion unified generator

![ChildDiffusion Pipeline](Figures/Figure%20Block%20Diagram.png)

---

## ⚙️ Configuration

Refer to [`ChildDiffusion_Boy_LoRA_Config.md`](ChildDiffusion_Boy_LoRA_Config.md) for a detailed breakdown of the training configuration used for the BoyDiffusion model.

---

## 📊 Inference Performance

Inference times for generating 800 images on an NVIDIA A6000 GPU (sampling steps = 22):

| Sampler   | Boy (min) | Girl (min) | Avg FPS |
|-----------|-----------|-------------|---------|
| DDIM      | 3.6       | 3.8         | ~3.6    |
| DPM++2M   | 4.5       | 4.6         | ~2.9    |
| Euler     | 3.5       | 3.6         | ~3.7    |
| Euler a   | 3.9       | 4.1         | ~3.3    |

---

## 📁 Dataset Release

We release a curated synthetic dataset of child faces across five ethnic groups with annotated variations in expression, head pose, and identity. All data is generated using ChildDiffusion and is compliant with ethical standards.

---

## 📜 License

This project is released under the MIT License.

---

## 👨‍💻 Citation & Contact

If you use this work, please cite our paper (coming soon). For questions or collaboration inquiries, please contact [Your Name] or open an issue.



---

## 🛠 Installation & Setup

### 🔧 Environment Setup

We recommend using Python 3.8+ with PyTorch (CUDA enabled). Create and activate a new environment:

```bash
conda create -n childdiffusion python=3.8
conda activate childdiffusion
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

Clone the DreamBooth-compatible repository (e.g., `diffusers`, `kohya-trainer`, or other GUI-based DreamBooth UIs):

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
pip install transformers accelerate safetensors datasets
```

---

## 📥 Download Base Model (Stable Diffusion v1.5)

You need to download the base Stable Diffusion v1.5 weights and place them in your DreamBooth directory:

```bash
# Download from Hugging Face
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

Once downloaded, place it in your training path under a folder like:

```
/your_dreambooth_path/stable-diffusion-v1-5/
```

---

## 🧪 Training with DreamBooth

To train the BoyDiffusion and GirlDiffusion models with concept-specific prompts:

```bash
accelerate launch train_dreambooth.py   --pretrained_model_name_or_path=./stable-diffusion-v1-5   --instance_data_dir=./data/boy_faces   --output_dir=./output/boy_diffusion   --instance_prompt="a portrait of boyface"   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=1   --learning_rate=1e-4   --lr_scheduler="constant"   --num_train_epochs=20
```

Repeat the process for GirlDiffusion by changing the data path and prompt.

---

## 🖼️ Sample Training Image Sets

### BoyDiffusion

![Boy Training Sample](Assets/Boy.png)

### GirlDiffusion

![Girl Training Sample](Assets/Girls.png)

Make sure your training images are consistent in framing and resolution (e.g., 512×512) for best results.

---

## 🚀 Inference

Once the models are trained, you can generate new images using:

```bash
python scripts/txt2img.py   --prompt "a smiling child wearing glasses, front-facing, soft lighting"   --ckpt ./output/childdiffusion_model.ckpt   --plms
```

Or via GUI-based tools like AUTOMATIC1111’s WebUI for DreamBooth.

---

