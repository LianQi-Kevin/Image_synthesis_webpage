import argparse
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import tqdm, trange
import logging


class text2img(object):
    def __init__(self, ckpt, config, n_rows=2, output_dir="outputs/txt2img-samples", precision="auto_cast"):
        # load_model
        self.ckpt = ckpt
        self.config = config
        self.model, self.device = self._load_model(verbose=False)

        # synthesis
        self.precision = precision
        self.n_iter = 4
        self.n_samples = 4
        self.n_rows = n_rows if n_rows > 0 else self.n_samples
        self.latent_channel = 4
        self.downsampling_factor = 8
        self.plms = False
        self.fixed_code = False
        self.scale = 5.0

        # output
        # # grid
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # # single
        self.sample_path = os.path.join(self.output_dir, "samples")
        os.makedirs(self.sample_path, exist_ok=True)
        self.base_count = len(os.listdir(self.sample_path))  # 单张图序号
        self.grid_count = len(os.listdir(self.output_dir)) - 1  # 组图序号

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
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        logging.info("Successful load {}".format(self.ckpt))
        return model, device

    def synthesis(self, prompt, img_H=256, img_W=256, seed=42,
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

        precision_scope = autocast if self.precision == "auto_cast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    all_samples = list()
                    for _ in trange(n_iter, desc="Sampling"):
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
                                                             # dynamic_threshold=self.dyn,
                                                             x_T=start_code)

                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            all_samples.append(x_samples_ddim)
        return all_samples

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


def make_args():
    parser = argparse.ArgumentParser("TEXT TO IMG")

    parser.add_argument("--prompt", type=str, nargs="?", default="a painting of a virus monster playing guitar",
                        help="the prompt to render")
    parser.add_argument("--out_dir", type=str, nargs="?", help="dir to write results to",
                        default="outputs/txt2img-samples")
    parser.add_argument("--skip_grid", action='store_true',
                        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples", )
    parser.add_argument("--skip_save", action='store_true',
                        help="do not save individual samples. For speed measurements.", )
    parser.add_argument("--ddim_steps", type=int, default=50, help="number of ddim sampling steps", )
    parser.add_argument("--plms", action='store_true', help="use plms sampling", )
    parser.add_argument("--fixed_code", action='store_true',
                        help="if enabled, uses the same starting code across all samples ", )
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="ddim eta (eta=0.0 corresponds to deterministic sampling)", )
    parser.add_argument("--n_iter", type=int, default=1, help="sample this often", )
    parser.add_argument("--H", type=int, default=256, help="image height, in pixel space", )
    parser.add_argument("--W", type=int, default=256, help="image width, in pixel space", )
    parser.add_argument("--C", type=int, default=4, help="latent channels", )
    parser.add_argument("--f", type=int, default=8, help="downsampling factor, most often 8 or 16", )
    parser.add_argument("--n_samples", type=int, default=8,
                        help="how many samples to produce for each given prompt. A.k.a batch size", )
    parser.add_argument("--n_rows", type=int, default=0, help="rows in the grid (default: n_samples)", )
    parser.add_argument("--scale", type=float, default=5.0,
                        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))", )
    parser.add_argument("--dyn", type=float,
                        help="dynamic thresholding from Imagen, in latent space (TODO: try in pixel space with intermediate decode)", )
    parser.add_argument("--from-file", type=str, help="if specified, load prompts from this file", )
    parser.add_argument("--config", type=str,
                        default="logs/f8-kl-clip-encoder-256x256-run1/configs/2022-06-01T22-11-40-project.yaml",
                        help="path to config which constructs model", )
    parser.add_argument("--ckpt", type=str, default="logs/f8-kl-clip-encoder-256x256-run1/checkpoints/last.ckpt",
                        help="path to checkpoint of model", )
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)", )
    parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "auto_cast"],
                        default="auto_cast")
    return parser.parse_args()


def main_class(opt):
    txt2img = text2img(opt.ckpt, opt.config, output_dir=opt.out_dir)
    print("Successful init class")
    tic_time = time.time()
    all_samples = txt2img.synthesis(opt.prompt, img_H=256, img_W=256, seed=np.random.randint(1, 1000000),
                                    n_samples=3, n_iter=3, ddim_steps=50, scale=5.0, ddim_eta=0.0)
    # all_samples = txt2img.synthesis(opt.prompt, img_H=384, img_W=1024, seed=2156486,
    #                                 n_samples=1, n_iter=1, ddim_steps=50, scale=5.0, ddim_eta=1.0)
    txt2img.save_img(all_samples, single_save=True, grid_save=True)
    print(time.time() - tic_time)


if __name__ == "__main__":
    # args
    opt = make_args()

    # ----------
    # 调试用 覆盖args
    opt.prompt = "sunset, sun rays showing through the woods in front, clear night sky, stars visible, mountain in the back, lake in front reflecting the night sky and mountain, photo realistic, 8K, ultra high definition, cinematic"
    opt.config = "/root/latent-Diffusion-Models/configs/latent-diffusion/txt2img-1p4B-eval.yaml"
    opt.ckpt = "/root/latent-Diffusion-Models/models/ldm/text2img-large/model.ckpt"
    opt.ddim_eta = 0.0  # 0.0 corresponds to deterministic sampling
    opt.n_samples = 3  # batch size    # X
    opt.n_iter = 3  # Y
    opt.scale = 5.0  # https://benanne.github.io/2022/05/26/guidance.html
    opt.ddim_steps = 50  # ddim sampling steps
    opt.out_dir = "outputs/txt2img-samples"  # output dir
    opt.H = 256  # Must be divisible by 8
    opt.W = 256  # Must be divisible by 8
    opt.skip_save = True  # skip save single img
    opt.skip_grid = False  # skip save grid img
    opt.seed = np.random.randint(1, 100)  # The same results when the seeds are the same
    # ----------

    main_class(opt)
