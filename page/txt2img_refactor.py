import argparse
import os
from contextlib import nullcontext
from itertools import islice

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


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
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
    return model


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
                        help="ddim eta (eta=0.0 corresponds to deterministic sampling", )
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


def load_model(opt):
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    return model, device


def text2img(opt, model, device):
    # 默认采用DDIM采样， 相较plms采样更平滑
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    batch_size = opt.n_samples

    # prompts
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision == "auto_cast" else nullcontext
    # 使用with代换try/finally
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                for _ in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         dynamic_threshold=opt.dyn,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        all_samples.append(x_samples_ddim)

    return all_samples


def postprocess(all_samples, opt, single=False):
    if single:
        images_list = list()
        for x_samples_ddim in all_samples:
            for x_sample in x_samples_ddim:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                images_list.append(Image.fromarray(x_sample.astype(np.uint8)))
        return images_list
    else:
        n_rows = opt.n_rows if opt.n_rows > 0 else opt.n_samples
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=n_rows)
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        return Image.fromarray(grid.astype(np.uint8))


def main():
    # args
    opt = make_args()

    # ----------
    # 调试用 覆盖args
    # opt.prompt = "sunset, sun rays showing through the woods in front, clear night sky, stars visible, mountain in the back, lake in front reflecting the night sky and mountain, photo realistic, 8K, ultra high definition, cinematic"
    opt.prompt = "A stained glass window of a panda eating bamboo, photo realistic, 8K, ultra high definition, cinematic"
    opt.config = "/root/latent-Diffusion-Models/configs/latent-diffusion/txt2img-1p4B-eval.yaml"
    opt.ckpt = "/root/latent-Diffusion-Models/models/ldm/text2img-large/model.ckpt"
    opt.ddim_eta = 0.0  # 0.0 corresponds to deterministic sampling
    opt.n_samples = 3   # batch size    # X
    opt.n_iter = 3      # Y
    opt.scale = 5.0     # https://benanne.github.io/2022/05/26/guidance.html
    opt.ddim_steps = 50     # ddim sampling steps
    opt.out_dir = "outputs/txt2img-samples"     # output dir
    opt.H = 256     # Must be divisible by 8
    opt.W = 256     # Must be divisible by 8
    opt.skip_save = True    # skip save single img
    opt.skip_grid = False   # skip save grid img
    opt.seed = np.random.randint(1, 100)    # The same results when the seeds are the same
    # ----------

    seed_everything(opt.seed)

    # load model
    model, device = load_model(opt)

    # output path
    os.makedirs(opt.out_dir, exist_ok=True)
    out_path = opt.out_dir

    # single img msg
    sample_path = os.path.join(out_path, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))   # 单张图序号
    grid_count = len(os.listdir(out_path)) - 1  # 组图序号

    all_samples = text2img(opt, model, device)

    # 单张图
    if not opt.skip_save:
        for img in postprocess(all_samples, opt, single=True):
            img.save(os.path.join(sample_path, f"{base_count:05}.png"))
            base_count += 1

    # 网格图
    if not opt.skip_grid:
        postprocess(all_samples, opt, single=False).save(os.path.join(out_path, f'grid-{grid_count:04}.png'))


if __name__ == "__main__":
    main()
