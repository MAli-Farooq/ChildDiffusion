girl_lora_content = """
# GirlDiffusion: LoRA Model Configuration Overview

This document provides a structured overview of the model tuning parameters used for training the **GirlDiffusion** pipeline using Low-Rank Adaptation (LoRA). The setup is based on Stable Diffusion v1.5 with concept-specific fine-tuning.

---

## 🧠 General Training Parameters

| Parameter               | Value / Description                            |
|-------------------------|------------------------------------------------|
| base_learning_rate      | 1.0e-04                                        |
| target                  | ldm.models.diffusion.ddpm.LatentDiffusion     |
| linear_start            | 0.00085                                        |
| linear_end              | 0.0120                                         |
| num_timesteps_cond      | 1                                              |
| log_every_t             | 200                                            |
| timesteps               | 1000                                           |
| image_size              | 64                                             |
| channels                | 4                                              |
| first_stage_key         | "image"                                        |
| cond_stage_key          | "caption"                                      |
| monitor                 | val/loss_simple_ema                            |
| scale_factor            | 0.18215                                        |
| use_ema                 | False                                          |

---

## ⏱ Scheduler Configuration

| Parameter               | Value / Description                            |
|-------------------------|------------------------------------------------|
| scheduler.target        | ldm.lr_scheduler.LambdaLinearScheduler         |
| warm_up_steps           | [10000]                                        |
| cycle_lengths           | [10000000000000]                               |
| f_start                 | [1.e-6]                                        |
| f_max                   | [1.0]                                          |
| f_min                   | [1.0]                                          |

---

## 🧩 UNet Configuration

| Parameter                   | Value / Description                         |
|-----------------------------|---------------------------------------------|
| image_size                  | 32                                           |
| in_channels                 | 4                                            |
| out_channels                | 4                                            |
| model_channels              | 320                                          |
| channel_mult                | [1, 2, 4, 4]                                 |
| attention_resolutions       | [4, 2, 1]                                    |
| num_res_blocks              | 2                                            |
| num_heads                   | 8                                            |
| use_spatial_transformer     | True                                         |
| transformer_depth           | 1                                            |
| context_dim                 | 768                                          |
| use_checkpoint              | True                                         |
| legacy                      | False                                        |

---

## 🎨 First Stage Autoencoder (VAE)

| Parameter                   | Value / Description                         |
|-----------------------------|---------------------------------------------|
| target                      | ldm.models.autoencoder.AutoencoderKL        |
| embed_dim                   | 4                                            |
| monitor                     | val/rec_loss                                 |
| double_z                    | True                                         |
| z_channels                  | 4                                            |
| resolution                  | 256                                          |
| in_channels                 | 3                                            |
| out_ch                      | 3                                            |
| ch                          | 128                                          |
| ch_mult                     | [1, 2, 4, 4]                                 |
| num_res_blocks              | 2                                            |
| attn_resolutions            | []                                           |
| dropout                     | 0.0                                          |
| lossconfig.target           | torch.nn.Identity                            |

---

## 🧾 Conditioning Stage

| Parameter                   | Value / Description                         |
|-----------------------------|---------------------------------------------|
| cond_stage_config.target    | ldm.modules.encoders.modules.FrozenCLIPEmbedder |
| cond_stage_trainable        | False                                       |
| conditioning_key            | crossattn                                   |
"""

# Save the markdown content to a file
girl_lora_path = Path("/mnt/data/GirlDiffusion_LoRA_Config.md")
girl_lora_path.write_text(girl_lora_content)

girl_lora_path.name
