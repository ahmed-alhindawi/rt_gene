"""
BSD 3-Clause License

Copyright (c) 2017, Adrian Bulat
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import cv2
import numpy as np

try:
    from .net_s3fd import s3fd
except ImportError:
    from net_s3fd import s3fd


class SFDDetector(object):
    __WHITENING = np.array([104, 117, 123])

    def __init__(self, device, path_to_detector=None):
        self.device = device

        if path_to_detector is None:
            import rospkg
            path_to_detector = rospkg.RosPack().get_path('rt_gene') + '/model_nets/SFD/s3fd_facedetector.pth'

        face_detector = s3fd()
        face_detector.load_state_dict(torch.load(path_to_detector))
        face_detector.eval()
        face_detector.to(device)

        ex_input = torch.randn((1, 3, 224, 224)).to(device).float()
        self.face_detector = torch.jit.trace(face_detector, ex_input)
        self.face_detector = torch.jit.optimize_for_inference(self.face_detector)

    def detect_from_image(self, image, threshold=0.8):
        # image = self.tensor_or_path_to_ndarray(tensor_or_path)
        bboxlist = self.detect(self.face_detector, image, device=self.device, process_device="cpu")

        keep = self.nms(bboxlist, 0.3)
        keep_t = torch.stack(keep, dim=0)
        bboxlist = bboxlist[keep_t, :]
        bbox_list_above_threshold = torch.nonzero(bboxlist[:, 4] >= threshold)
        bboxlist = bboxlist[bbox_list_above_threshold].view(-1, 5)
        return bboxlist

    @staticmethod
    def tensor_or_path_to_ndarray(tensor_or_path, rgb=True):
        """Convert path (represented as a string) or torch.tensor to a numpy.ndarray

        Arguments:
            tensor_or_path {numpy.ndarray, torch.tensor or string} -- path to the image, or the image itself
        """
        if isinstance(tensor_or_path, str):
            from skimage import io
            return cv2.imread(tensor_or_path) if not rgb else io.imread(tensor_or_path)
        elif torch.is_tensor(tensor_or_path):
            # Call cpu in case its coming from cuda
            return tensor_or_path.cpu().numpy()[..., ::-1].copy() if not rgb else tensor_or_path.cpu().numpy()
        elif isinstance(tensor_or_path, np.ndarray):
            return tensor_or_path[..., ::-1].copy() if not rgb else tensor_or_path
        else:
            raise TypeError

    @staticmethod
    def nms(dets, thresh):
        if 0 == len(dets):
            return []
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort().flip(dims=(0,))

        keep = []
        while order.numel() > 0:
            bb_current = order[0]
            bb_others = order[1:]
            keep.append(bb_current)

            xx1 = torch.maximum(x1[bb_current], x1[bb_others])
            yy1 = torch.maximum(y1[bb_current], y1[bb_others])
            xx2 = torch.minimum(x2[bb_current], x2[bb_others])
            yy2 = torch.minimum(y2[bb_current], y2[bb_others])

            xx_diff = xx2 - xx1 + 1
            yy_diff = yy2 - yy1 + 1
            w, h = torch.maximum(torch.tensor(0.0), xx_diff), torch.maximum(torch.tensor(0.0), yy_diff)
            ovr = w * h / (areas[bb_current] + areas[bb_others] - w * h)

            inds = torch.nonzero(ovr <= thresh).flatten()
            order = order[inds + 1]

        return keep

    def detect(self, net, img, device, process_device="cpu"):
        img = img - self.__WHITENING
        img = torch.Tensor(img).float().to(device)
        img = img.permute((2, 0, 1)).unsqueeze(0)

        with torch.no_grad():
            nn_out = net(img)

        nn_out = [x.to(process_device) for x in nn_out]
        oreg_nn = [nn_out[1], nn_out[3], nn_out[5], nn_out[7], nn_out[9], nn_out[11]]
        ocls_nn = [nn_out[0], nn_out[2], nn_out[4], nn_out[6], nn_out[8], nn_out[10]]

        bboxlist = list()
        for i, (ocls, oreg) in enumerate(zip(ocls_nn, oreg_nn)):
            ocls = ocls.softmax(dim=1)
            stride = 2 ** (i + 2)

            poss = (ocls[:, 1, :, :] > 0.05).nonzero()
            hindex = poss[:, 1]
            windex = poss[:, 2]
            axc = stride / 2 + windex * stride
            ayc = stride / 2 + hindex * stride
            score = ocls[0, 1, hindex, windex]
            loc = oreg[0, :, hindex, windex].T
            priors = torch.vstack([axc, ayc, torch.full((len(axc),), stride * 4.0).to(process_device),
                                   torch.full((len(axc),), stride * 4.0).to(process_device)]).T
            variances = [0.1, 0.2]
            boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                               priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
            boxes[:, :2] -= boxes[:, 2:] / 2
            boxes[:, 2:] += boxes[:, :2]
            box = torch.hstack([boxes, score.unsqueeze(1)])
            bboxlist.append(box)

        bboxlist = torch.vstack(bboxlist)
        if 0 == len(bboxlist):
            bboxlist = torch.zeros((1, 5))

        return bboxlist
