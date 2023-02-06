import argparse
import os
import time
import logging
from contextlib import nullcontext

import numpy as np
import torch
import PIL
from PIL import Image
from einops import rearrange, repeat
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import tqdm, trange


class all2img(object):
    def __init__(self, ckpt, config, n_rows=2, output_dir="outputs", precision="autocast"):
        # load model
        self.ckpt = ckpt
        self.config = config
        self.model, self.device = self._load_model(verbose=False)

        # synthesis
        self.precision_scope = autocast if precision == "autocast" else nullcontext
        self.n_iter = 4
        self.n_samples = 4
        self.n_rows = n_rows if n_rows > 0 else self.n_samples
        self.latent_channel = 4
        self.downsampling_factor = 8
        self.plms = False
        self.fixed_code = False
        self.scale = 5.0
        self.strength = 0.8

        # output
        # # grid
        self.output_dir = output_dir
        # # single
        self.sample_path = os.path.join(self.output_dir, "samples")
        os.makedirs(self.sample_path, exist_ok=True)
        self.base_count = len(os.listdir(self.sample_path))  # 单张图序号
        self.grid_count = len(os.listdir(self.output_dir)) - 1  # 组图序号

    # stable diffusion model load
    def _load_model(self, verbose=False):
        config = OmegaConf.load(self.config)
        logging.info("Loading model from {}".format(self.ckpt))
        pl_sd = torch.load(self.ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            logging.info("Global Step: {}".format(pl_sd['global_step']))
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
        model.cuda()
        model.eval()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        logging.info("Successful load {}".format(self.ckpt))
        return model, device

    # img2img init image load
    def _load_init_img(self, img_path: str, batch_size: int):
        assert os.path.exists(img_path), "{} not found".format(img_path)
        # preprocess img
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        logging.info("loaded input image of size ({}, {}) from {}".format(w, h, img_path))
        w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        # make grid
        init_image = (2. * image - 1.).to(self.device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(init_image))  # move to latent space
        return init_latent

    # export image save
    def save_img(self, all_samples, single_save=True, grid_save=True, n_rows=None):
        if single_save:
            for img in iter(self.postprocess(all_samples, single=True)):
                single_img_path = os.path.join(self.sample_path, f"{self.base_count:05}.png")
                img.save(single_img_path)
                print("Successful save {}".format(single_img_path))
                self.base_count += 1
        if grid_save:
            if n_rows is not None:
                self.n_rows = n_rows
            for img in iter(self.postprocess(all_samples, single=False)):
                grid_img_path = os.path.join(self.output_dir, f'grid-{self.grid_count:04}.png')
                img.save(grid_img_path)
                print("Successful save {}".format(grid_img_path))
                self.grid_count += 1

        if single_save and grid_save:
            return self.postprocess(all_samples, single=False)
        elif not single_save and not grid_save:
            return self.postprocess(all_samples, single=False)
        elif not single_save and grid_save:
            return self.postprocess(all_samples, single=False)
        else:
            return self.postprocess(all_samples, single=True)

    # model output postprocess
    def postprocess(self, all_samples, single=False):
        if single:
            images_list = list()
            for x_samples_ddim in all_samples:
                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    images_list.append(Image.fromarray(x_sample.astype(np.uint8)))
            return images_list
        else:
            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=self.n_rows)
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            return [Image.fromarray(grid.astype(np.uint8))]

    # text2img synthesis
    def text2img(self, prompt: str, img_H=256, img_W=256, seed=42,
                 n_samples=4, n_iter=4, ddim_steps=50, scale=None, ddim_eta=0.0,
                 plms=None, fixed_code=None, latent_channel=None, downsampling_factor=None):

        if plms is None:
            plms = self.plms
        if fixed_code is None:
            fixed_code = self.fixed_code
        if latent_channel is None:
            latent_channel = self.latent_channel
        if downsampling_factor is None:
            downsampling_factor = self.downsampling_factor
        if scale is None:
            scale = self.scale

        self.n_samples = n_samples
        self.n_iter = n_iter

        seed_everything(seed)

        # 默认采用DDIM采样， 相较plms采样更平滑
        if plms:
            sampler = PLMSSampler(self.model)
        else:
            sampler = DDIMSampler(self.model)

        batch_size = n_samples

        # prompts
        assert prompt is not None
        data = [batch_size * [prompt]]

        start_code = None
        if fixed_code:
            start_code = torch.randn(
                [n_samples, latent_channel, img_H // downsampling_factor, img_W // downsampling_factor],
                device=self.device)

        with torch.no_grad():
            with self.precision_scope("cuda"):
                with self.model.ema_scope():
                    all_samples = list()
                    for _ in trange(self.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if scale != 1.0:
                                uc = self.model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)
                            shape = [latent_channel, img_H // downsampling_factor, img_W // downsampling_factor]
                            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                             conditioning=c,
                                                             batch_size=n_samples,
                                                             shape=shape,
                                                             verbose=False,
                                                             unconditional_guidance_scale=scale,
                                                             unconditional_conditioning=uc,
                                                             eta=ddim_eta,
                                                             x_T=start_code)

                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            all_samples.append(x_samples_ddim)
        return all_samples

    # img2img synthesis
    def img2img(self, prompt, init_img, seed=42, n_samples=4, n_iter=4,
                ddim_steps=50, scale=None, ddim_eta=0.0, strength=None, plms=None):
        if plms is None:
            plms = self.plms
        if scale is None:
            scale = self.scale
        if strength is None:
            strength = self.strength

        self.n_samples = n_samples
        self.n_iter = n_iter

        seed_everything(seed)

        # 默认采用DDIM采样， 相较plms采样更平滑
        if plms:
            sampler = PLMSSampler(self.model)
        else:
            sampler = DDIMSampler(self.model)

        sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

        batch_size = n_samples

        # load init image
        init_latent = self._load_init_img(init_img, batch_size)

        # prompts
        assert prompt is not None
        data = [batch_size * [prompt]]

        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(strength * ddim_steps)
        logging.info(f"target t_enc is {t_enc} steps")

        with torch.no_grad():
            with self.precision_scope("cuda"):
                with self.model.ema_scope():
                    all_samples = list()
                    for _ in trange(self.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if scale != 1.0:
                                uc = self.model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)

                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(init_latent,
                                                              torch.tensor([t_enc] * batch_size).to(self.device))
                            # decode it
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=uc, )

                            x_samples = self.model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                            all_samples.append(x_samples)
        return all_samples
