import glob
import os
import shutil
import json

import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt
import torch.nn.functional as F
# import fairseq


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def attr_dict(config_file):
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    return h

# def get_fairseq_model(h, device):
#     cp_path = h.fairseq_checkpoint_path
#     fairseq_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([
#         cp_path])
#     fairseq_model = fairseq_model[0]
#     # move model to GPU
#     fairseq_model.eval().to(device)
    
#     return fairseq_model

# def fairseq_loss(output, gt, fairseq_model):
#     """
#     fairseq feature mse loss, based on https://arxiv.org/abs/2301.04388
#     """
#     gt = gt.squeeze(1)
#     output = output.squeeze(1)
#     gt_f = fairseq_model.feature_extractor(gt)
#     output_f = fairseq_model.feature_extractor(output)
#     mse_loss = F.mse_loss(gt_f, output_f)
#     return mse_loss


# def speaker_loss(output, gt_embed, embedder):
#     """
#     """
#     output = output.squeeze(1)
#     output_embed = embedder.generate_speaker_embedding_torch_batched(output)
#     # speaker_loss = F.mse_loss(gt_embed, output_embed.squeeze(1))
#     speaker_loss = F.cosine_embedding_loss(gt_embed, output_embed.squeeze(1), torch.ones(gt_embed.size(0)).to(gt_embed.device))
#     return speaker_loss