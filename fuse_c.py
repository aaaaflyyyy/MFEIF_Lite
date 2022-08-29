import pathlib
import statistics
import time

import cv2
import kornia
import torch
from tqdm import tqdm

from functions.feather_fuse import FeatherFuse
from models.attention import Attention
from models.constructor_c import Constructor
from models.extractor_c import Extractor

# minimize model
from models.constructor_minimize import Constructor as Constructor_lite
from models.extractor_minimize import Extractor as Extractor_lite
from models.extractor_minimize_tradConv import Extractor as Extractor_lite_tradConv

class Fuse:
    """
    fuse with infrared folder and visible folder
    """

    def __init__(self, model_path: str, USE_liteModel: bool, USE_tradCONV: bool, USE_Attention: bool):
        """
        :param model_path: path of pre-trained parameters
        """

        # device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # model parameters
        params = torch.load(model_path, map_location='cpu')

        if USE_liteModel:
            if USE_tradCONV:
                net_ext = Extractor_lite_tradConv()
            else:
                net_ext = Extractor_lite()

            net_con = Constructor_lite()
        else:
            net_ext = Extractor()
            net_con = Constructor()

        # load extractor
        net_ext.load_state_dict(params['ext'])
        net_ext.to(device)
        net_ext.eval()
        self.net_ext = net_ext

        # load constructor
        net_con.load_state_dict(params['con'])
        net_con.to(device)
        net_con.eval()
        self.net_con = net_con

        # load attention layer
        if USE_Attention:
            net_att = Attention()
            net_att.load_state_dict(params['att'])
            net_att.to(device)
            net_att.eval()
            self.net_att = net_att
        else:
            self.net_att = None

        # softmax and feather-fuse
        self.softmax = torch.nn.Softmax(dim=1)
        self.feather_fuse = FeatherFuse()

    def __call__(self, ir_folder: str, vi_folder: str, dst: str):
        """
        fuse with ir folder and vi folder and save fusion image into dst
        :param ir_folder: infrared image folder
        :param vi_folder: visible image folder
        :param dst: fusion image output folder
        """

        # ext_params = sum(p.numel() for p in self.net_ext.parameters())
        # con_params = sum(p.numel() for p in self.net_con.parameters())
        # if self.net_att:
        #     att_params = sum(p.numel() for p in self.net_att.parameters())
        # else:
        #     att_params = 0

        # print('ext_params: ', ext_params)
        # print('con_params: ', con_params)
        # print('att_params: ', att_params)
        # print('total_params: ', ext_params+con_params+att_params)

        # image list
        ir_folder = pathlib.Path(ir_folder)
        vi_folder = pathlib.Path(vi_folder)
        ir_list = [x for x in ir_folder.glob('*') if x.suffix in ['.bmp', '.png', '.jpg']][:200]
        vi_list = [x for x in vi_folder.glob('*') if x.suffix in ['.bmp', '.png', '.jpg']][:200]

        # check image name and fuse
        fuse_time = []
        rge = tqdm(zip(ir_list, vi_list))

        for ir_path, vi_path in rge:
            start = time.time()

            # check image name
            ir_name = ir_path.stem
            vi_name = vi_path.stem
            rge.set_description(f'fusing {ir_name}')
            assert ir_name == vi_name

            # read image
            ir = self._imread(str(ir_path)).unsqueeze(0)
            vi = self._imread(str(vi_path)).unsqueeze(0)
            ir = ir.to(self.device)
            vi = vi.to(self.device)

            # network forward
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            fu = self._forward(ir, vi)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            

            # save fusion tensor
            fu_path = pathlib.Path(dst, ir_path.name)
            self._imsave(fu_path, fu)

            end = time.time()
            fuse_time.append(end - start)
        
        # time analysis
        if len(fuse_time) > 2:
            std = statistics.stdev(fuse_time[1:])
            mean = statistics.mean(fuse_time[1:])
            print(f'fuse std time: {std:.2f}')
            print(f'fuse avg time: {mean:.2f}')
            print('fps (equivalence): {:.2f}'.format(1. / mean))
        else:
            print(f'fuse avg time: {fuse_time[0]:.2f}')


    @torch.no_grad()
    def _forward(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        ir_1 = self.net_ext(ir)
        vi_1 = self.net_ext(vi)

        if self.net_att:
            ir_att = self.net_att(ir)
            vi_att = self.net_att(vi)

            fus_1 = ir_1 * ir_att + vi_1 * vi_att
        else:
            fus_1 = ir_1 + vi_1


        # fus_1 = self.softmax(fus_1)
        fus_2 = self.net_con(fus_1/2)
        
        return fus_2

    @staticmethod
    def _imread(path: str, flags=cv2.IMREAD_GRAYSCALE) -> torch.Tensor:
        im_cv = cv2.imread(path, flags)
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.0).type(torch.FloatTensor)
        return im_ts

    @staticmethod
    def _imsave(path: pathlib.Path, image: torch.Tensor):
        im_ts = image.squeeze().cpu()
        path.parent.mkdir(parents=True, exist_ok=True)
        im_cv = kornia.utils.tensor_to_image(im_ts) * 255.
        cv2.imwrite(str(path), im_cv)


if __name__ == '__main__':

    model = 'MFEIF_c_woAttn'
    f = Fuse(f'./cache/{model}/best.pth', USE_liteModel=False, USE_tradCONV=False, USE_Attention=False)
    # f('data/mini_LLVIP/infrared', 'data/mini_LLVIP/visible', f'runs/mini_LLVIP/{model}')
    # f('data/TNO/Nato/thermal', 'data/TNO/Nato/visual', f'runs/TNO/Nato/{model}')
    f('data/test/ir', 'data/test/vi', f'runs/test/{model}')
