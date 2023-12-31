### code for evaluation of light estimation ###

import argparse
import os
import urllib

from probe_prediction import Model
#import ProbeModel
import numpy as np
import torch as th

import lightEstim

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('scene')
  parser.add_argument('--light_dir', type=int)
  parser.add_argument('-o', '--out')

  parser.add_argument("--cropx", type=int, default=512)
  parser.add_argument("--cropy", type=int, default=256)

  parser.add_argument("--cropsz", type=int, default=512)
  parser.add_argument("--checkpoint", required=False)
  parser.add_argument("--mip", type=int, default=2)
  parser.add_argument("--chromesz", type=int, default=64)
  parser.add_argument("--fmaps1", type=int, default=6)
  parser.add_argument("--gpu_id", type=int, default=0)
  opts,unknown = parser.parse_known_args()

  if opts.checkpoint is None:
    opts.checkpoint = \
      "checkpoints/probe_predict/t_1547304767_nsteps_000050000.checkpoint"
    lightEstim.ensure_checkpoint_downloaded(opts.checkpoint)

  model = SingleImageEvaluator(opts)

  I = lightEstim.query_images(opts.scene, mip=opts.mip,
      dirs=opts.light_dir)[0,0]

  # extract crop
  ox = opts.cropx
  oy = opts.cropy
  cropnp = I[oy:oy+opts.cropsz, ox:ox+opts.cropsz]

  # pre process input
  crop = np.moveaxis(cropnp, 2, 0)
  crop = crop / 255 - 0.5

  # run model
  pred = model.forward(crop)

  # post process prediction
  pred = np.moveaxis(pred, 0, 2) + 0.5
  pred = (np.clip(autoexpose(pred), 0, 1) * 255).astype('uint8')

  # write results
  lightEstim.writejpg(cropnp, "%s.input.jpg" % opts.out)
  lightEstim.writejpg(pred, "%s.pred.jpg" % opts.out)




def autoexpose(I):
  n = np.percentile(I[:,:,1], 90)
  if n > 0:
    I = I / n
  return I

class SingleImageEvaluator:
  def __init__(self, opts):
    self.net = ProbeModel.make_net_fully_convolutional(
        chromesz=opts.chromesz,
        fmaps1=opts.fmaps1,
        xysize=opts.cropsz)
    self.net.cuda()
    self.net.load_state_dict(th.load(opts.checkpoint,map_location='cuda:0'))

  def forward(self, image):
    """Accept (3,512,512) image in [-0.5; 0.5] range.

    Returns (3,64,64) image in [-0.5; 0.5] range.
    """
    x = image
    h, w = image.shape[-2:]
    x = th.Tensor(x).view(1, 3, h, w)
    
    self.net.zero_grad()
    with th.no_grad():
      pred = self.net(th.autograd.Variable(x).cuda())

    pred = pred.view(3, self.net.chromesz, self.net.chromesz)
    pred = pred.detach().cpu().numpy()
    return pred


if __name__ == "__main__":
  main()
