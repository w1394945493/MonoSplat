import os
import os.path as osp
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import json
import random
import pickle as pkl
from functools import cached_property
from pathlib import Path
import imageio.v2 as imageio
import glob
import torch
import torch.nn.functional as F
import PIL
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import cv2
import copy
from io import BytesIO
from einops import rearrange, repeat, einsum

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# from model.utils.image import resize_image, HWC3
# from model.utils.typing import *
# from model.utils.camera import get_camera, rescale_intrisic
# from model.utils.ops import get_cam_info_gaussian, get_ray_directions, get_rays

from .utils.image import resize_image, HWC3
from .utils.typing import *
from .utils.camera import get_camera, rescale_intrisic
from .utils.ops import get_cam_info_gaussian, get_ray_directions, get_rays


from dataclasses import dataclass

@dataclass
class DatasetNuScencesCfg:
    name: Literal["nuscences"] # todo 必须是"nuscences"字段
    background_color: list[float]
    resolution: list[int]
    use_center: bool
    use_first: bool
    use_last: bool
    near:float
    far:float



bins_demo = [
'scene04219bfdc9004ba2af16d3079ecc4353_bin061',
'scene07aed9dae37340a997535ad99138e243_bin058',
'scene0ac05652a4c44374998be876ba5cd6fd_bin121',
'scene16e50a63b809463099cb4c378fe0641e_bin231',
# 'scene197a7e4d3de84e57af17b3d65fcb3893_bin177',
# 'scene19d97841d6f64eba9f6eb9b6e8c257dc_bin001',
# 'scene201b7c65a61f4bc1a2333ea90ba9a932_bin071',
# 'scene2086743226764f268fe8d4b0b7c19590_bin043',
# 'scene265f002f02d447ad9074813292eef75e_bin128',
# 'scene26a6b03c8e2f4e6692f174a7074e54ff_bin103',
# 'scene2abb3f3517c64446a5768df5665da49d_bin128',
# 'scene2ca15f59d656489a8b1a0be4d9bead4e_bin003',
# 'scene2ed0fcbfc214478ca3b3ce013e7723ba_bin154',
# 'scene2f56eb47c64f43df8902d9f88aa8a019_bin136',
# 'scene3045ed93c2534ec2a5cabea89b186bd9_bin176',
# 'scene36f27b26ef4c423c9b79ac984dc33bae_bin207',
# 'scene3a2d9bf6115f40898005d1c1df2b7282_bin107',
# 'scene3ada261efee347cba2e7557794f1aec8_bin005',
# 'scene3dd2be428534403ba150a0b60abc6a0a_bin083',
# 'scene3dd9ad3f963e4f588d75c112cbf07f56_bin131',
# 'scene3f90afe9f7dc49399347ae1626502aed_bin095',
# 'scene4962cb207a824e57bd10a2af49354b16_bin089',
# 'scene5301151d8b6a42b0b252e95634bd3995_bin121',
# 'scene5521cd85ed0e441f8d23938ed09099dd_bin067',
# 'scene6a24a80e2ea3493c81f5fdb9fe78c28a_bin033',
# 'scene6af9b75e439e4811ad3b04dc2220657a_bin115',
# 'scene7061c08f7eec4495979a0cf68ab6bb79_bin180',
# 'scene7365495b74464629813b41eacdb711af_bin067',
# 'scene76ceedbcc6a54b158eba9945a160a7bc_bin063',
# 'scene7e8ff24069ff4023ac699669b2c920de_bin014',
# 'scene813213458a214a39a1d1fc77fa52fa34_bin040',
# 'scene848ac962547c4508b8f3b0fcc8d53270_bin023',
# 'scene85651af9c04945c3a394cf845cb480a6_bin017',
# 'scene8edbc31083ab4fb187626e5b3c0411f7_bin017',
# 'scene9088db17416043e5880a53178bfa461c_bin005',
# 'scene91c071bcc1ad4fa1b555399e1cfbab79_bin002',
# 'scene91f797db8fb34ae5b32ba85eecae47c9_bin004',
# 'scene9709626638f5406f9f773348150d02fd_bin092',
# 'sceneafbc2583cc324938b2e8931d42c83e6b_bin009',
# 'sceneb07358651c604e2d83da7c4d4755de73_bin017',
# 'sceneb94fbf78579f4ff5ab5dbd897d5e3199_bin155',
# 'scenecba3ddd5c3664a43b6a08e586e094900_bin032',
# 'scened3b86ca0a17840109e9e049b3dd40037_bin040',
# 'scenee036014a715945aa965f4ec24e8639c9_bin005',
# 'sceneefa5c96f05594f41a2498eb9f2e7ad99_bin092',
# 'scenef97bf749746c4c3a8ad9f1c11eab6444_bin009',

] # todo 一个bin对应一个场景

# from .transforms.loading import load_info, load_conditions
from .utils.loading import load_info, load_conditions








class nuScenesDataset(Dataset):
    # data_root: str = "data/nuScenes"
    data_root: str = "/home/lianghao/wangyushen/data/wangyushen/Datasets/dataset_omniscene" #! 修改为自己存放数据集的路径

    data_version: str = "interp_12Hz_trainval"
    #data_version: str = "v1.0-trainval"
    dataset_prefix: str = "/datasets/nuScenes" # .pkl文件中默认图片的根目录
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    camera_types_first = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT"
    ]
    camera_types_last = [
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(
        self,
        resolution: list = [224, 400],
        split: str = "train",
        use_center: bool = True,
        use_first: bool = False,
        use_last: bool = False,
        near = 0.5,
        far = 100.,
    ):

        super().__init__()

        self.reso = resolution
        self.use_center = use_center
        self.use_first = use_first
        self.use_last = use_last

        # todo (wys 10.23)
        self.near = near
        self.far = far

        # load bin tokens
        if split == "train":
            #for training
            self.bin_tokens = json.load(open(osp.join(self.data_root, self.data_version, "bins_train_3.2m.json")))["bins"] # todo len -> 135932
        elif split == "val":
            # for visualization during training
            self.bin_tokens = json.load(open(osp.join(self.data_root, self.data_version, "bins_val_3.2m.json")))["bins"] # todo len -> 30080
            self.bin_tokens = self.bin_tokens[:30000:3000][:10] # todo 取10个元素
        elif split == "test":
            # for evaluation
            self.bin_tokens = json.load(open(osp.join(self.data_root, self.data_version, "bins_val_3.2m.json")))["bins"] # todo len -> 30080
            # mini test
            #self.bin_tokens = self.bin_tokens[0::14][:2048]
        elif split == "demo": # todo 跑demo.py脚本时用
            # super mini test
            self.bin_tokens = bins_demo

        self.split = split # todo: __getitem__中，未做区分

    # todo (wys 10.23) 参考re10k
    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def __len__(self):
        return len(self.bin_tokens)

    def __getitem__(self, index):

        bin_token = self.bin_tokens[index]
        with open(osp.join(self.data_root, self.data_version, "bin_infos_3.2m", bin_token + ".pkl"), "rb") as f:
            bin_info = pkl.load(f)

        # todo: 多传感器帧级元信息
        sensor_info_center = {sensor: bin_info["sensor_info"][sensor][0] for sensor in self.camera_types + ["LIDAR_TOP"]}
        # sensor_info_center.keys(): 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'LIDAR_TOP'
        # todo：相机在该帧中的完整信息
        # sensor_info_center['CAM_FRONT'].keys(): 'data_path', 'type', 'sample_data_token',
        # 'data_path': 图像文件路径；'type'：数据类型，如"CAM_FRONT"；'sample_data_token'：NuScenes中用于索引该传感器数据的唯一ID
        # todo：字段变换相关信息
        # 'sensor2ego_translation', 'sensor2ego_rotation', 'ego2global_translation', 'ego2global_rotation', 'timestamp',
        # 'sensor2ego_translation'：传感器坐标系 -> 自车坐标系的平移向量[x,y,z] 'sensor2ego_rotation': 传感器坐标系 -> 自车坐标系旋转向量[qw,qx,qy,qz]
        # 'ego2global_translation'：自车坐标系 → 全局坐标系的平移。'ego2global_rotation': 自车坐标系 → 全局坐标系的旋转（四元数）。
        # 'sensor2lidar_rotation', 'sensor2lidar_translation', 'sensor2lidar_transform', 'sample_token', 'camera_intrinsics'
        # 'sensor2lidar_rotation'：传感器坐标系 → 雷达坐标系的旋转。'sensor2lidar_translation'：传感器坐标系 → 雷达坐标系的平移。'sensor2lidar_transform'：综合上述旋转和平移，形成齐次变换矩阵，用于点或射线坐标转换。
        '''
        # 可视化相机和自车的位置关系(俯视图)
        import matplotlib.pyplot as plt
        import numpy as np
        from pyquaternion import Quaternion

        # 相机名称
        camera_names = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        # 读取位置与朝向
        positions, directions = [], []
        for cam in camera_names:
            info = sensor_info_center[cam]
            t = np.array(info['sensor2ego_translation'])
            q = Quaternion(info['sensor2ego_rotation'])  # w, x, y, z
            forward = q.rotate(np.array([0, 0, 1]))  # 相机朝向
            positions.append(t)
            directions.append(forward)

        positions = np.array(positions)
        directions = np.array(directions)

        # 绘图
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("Top-down View of Cameras (Ego Vehicle Centered)", fontsize=13)
        ax.set_xlabel("X (forward)")
        ax.set_ylabel("Y (left)")
        ax.grid(True)
        ax.axis('equal')

        # 绘制车辆轮廓（假设车长4.5m，宽2m）
        car_length, car_width = 4.5, 2.0
        car_outline = np.array([
            [ car_length/2,  car_width/2],
            [ car_length/2, -car_width/2],
            [-car_length/2, -car_width/2],
            [-car_length/2,  car_width/2],
            [ car_length/2,  car_width/2]
        ])
        ax.plot(car_outline[:, 0], car_outline[:, 1], 'k-', linewidth=2)

        # 绘制相机位置与朝向
        for i, cam in enumerate(camera_names):
            x, y = positions[i][0], positions[i][1]
            dx, dy = directions[i][0], directions[i][1]
            ax.scatter(x, y, s=50, label=cam)
            ax.arrow(x, y, dx*0.5, dy*0.5, head_width=0.2, head_length=0.3, fc='r', ec='r')

        # 自车中心
        ax.scatter(0, 0, c='k', s=100, marker='x', label='Ego Vehicle')

        ax.legend()
        plt.savefig("ego_cameras_topdown.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        '''
        sensor_info_first = {sensor: bin_info["sensor_info"][sensor][1] for sensor in self.camera_types_first + ["LIDAR_TOP"]}
        sensor_info_last = {sensor: bin_info["sensor_info"][sensor][2] for sensor in self.camera_types_last + ["LIDAR_TOP"]}

        # =================== Input views of this bin ===================== #
        input_img_paths, input_c2ws, input_w2cs = [], [], []
        if self.use_center: # todo: 默认为True
            for cam in self.camera_types: # ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
                info = copy.deepcopy(sensor_info_center[cam])
                img_path, c2w, w2c = load_info(info) # c2w/w2c:(4,4)
                img_path = img_path.replace(self.dataset_prefix, self.data_root)
                input_img_paths.append(img_path) # 图像路径列表
                input_c2ws.append(c2w) # 相机矩阵列表
                input_w2cs.append(w2c)
        if self.use_first: # todo: 默认为False
            for cam in self.camera_types_first:
                info = copy.deepcopy(sensor_info_first[cam])
                img_path, c2w, w2c = load_info(info)
                img_path = img_path.replace(self.dataset_prefix, self.data_root)
                input_img_paths.append(img_path)
                input_c2ws.append(c2w)
                input_w2cs.append(w2c)
        if self.use_last: # todo: 默认为False
            for cam in self.camera_types_last:
                info = copy.deepcopy(sensor_info_last[cam])
                img_path, c2w, w2c = load_info(info)
                img_path = img_path.replace(self.dataset_prefix, self.data_root)
                input_img_paths.append(img_path)
                input_c2ws.append(c2w)
                input_w2cs.append(w2c)
        input_c2ws = torch.as_tensor(input_c2ws, dtype=torch.float32) # todo (n,4,4) n=6: 6个相机
        input_w2cs = torch.as_tensor(input_w2cs, dtype=torch.float32)
        # todo：加载图像并调整内参
        # load and modify images (cropped or resized if necessary), and modify intrinsics accordingly
        # todo input_imgs: 各片段的中心帧包含6张环视图，作为输入视图
        input_imgs, input_depths, input_depths_m, input_confs_m, input_cks = \
                    load_conditions(input_img_paths, self.reso) # todo：返回 图像 (6,3,h,w)、深度图、置信度 以及 相机内参矩阵
        input_cks = torch.as_tensor(input_cks, dtype=torch.float32) # todo (6,3,3) 相机内参矩阵
        # todo：从各个相机内参矩阵中提取焦距和主点: fx/fy: 水平/垂直方向焦距(单位：像素) cx/cy: 主点坐标，即图像中心在像素平面上的位置(单位：像素)
        input_fxs, input_fys, input_cxs, input_cys = input_cks[:, 0, 0], input_cks[:, 1, 1], input_cks[:, 0, 2], input_cks[:, 1, 2]

        # compute image fovs and pixel directions
        input_fovxs, input_fovys = [], []
        input_directions = []
        for fx, fy, cx, cy in zip(input_fxs, input_fys, input_cxs, input_cys):
            # todo：计算像素方向和视场角
            # todo: 像素方向：每个像素在图像平面上对应一条从相机中心出发的光线，像素
            direction = get_ray_directions(self.reso[0], self.reso[1],
                                           focal=[fx, fy], principal=[cx, cy]) # todo：get_ray_directions(): 计算每个像素的光线方向
            fovx = 2 * np.arctan(cx / fx)
            fovy = 2 * np.arctan(cy / fy) # todo: 视场角计算
            input_fovxs.append(fovx)
            input_fovys.append(fovy)
            input_directions.append(direction)
        input_fovxs = torch.as_tensor(input_fovxs, dtype=torch.float32)
        input_fovys = torch.as_tensor(input_fovys, dtype=torch.float32)

        input_directions = torch.stack(input_directions)
        # todo：根据相机外参计算光线：即将像素光线方向从相机坐标系变换到世界坐标系
        input_rays_o, input_rays_d = get_rays(
            input_directions, input_c2ws, keepdim=True, normalize=False) # todo 返回值 input_rays_o: 光线起点；input_rays_d: 光线方向
        # todo: 生成世界坐标 -> 像素坐标的变换矩阵
        # prepare w2i for volume-gs
        input_w2is = []
        for w2c, ck in zip(input_w2cs, input_cks):
            viewpad = torch.eye(4)
            viewpad[:ck.shape[0], :ck.shape[1]] = ck
            w2i = (viewpad @ w2c.T)
            input_w2is.append(w2i)
        input_w2is = torch.stack(input_w2is)

        # todo：准备输出视图(非关键帧) 用于渲染损失
        # ======= Render views from non-key frames for rendering losses ====== #
        output_img_paths, output_c2ws, output_w2cs = [], [], []

        frame_num = len(bin_info["sensor_info"]["LIDAR_TOP"])
        assert frame_num >= 3, "only got {} frames for bin{}".format(frame_num, bin_token) # todo 帧数应大于3，第0帧作为输入，第1、2帧作为输出真值监督
        if self.use_center:
            rend_indices = [[1, 2]] * 6
        else:
            rend_indices = [[0]] * 6

        for cam_id, cam in enumerate(self.camera_types): # todo camera_types: 每一帧有6张环视图
            indices = rend_indices[cam_id]
            for ind in indices:
                info = copy.deepcopy(bin_info["sensor_info"][cam][ind])
                img_path, c2w, w2c = load_info(info)
                img_path = img_path.replace(self.dataset_prefix, self.data_root)
                output_img_paths.append(img_path)
                output_c2ws.append(c2w)
                output_w2cs.append(w2c)
        output_c2ws = torch.as_tensor(output_c2ws, dtype=torch.float32)

        # load and modify images (cropped or resized if necessary), and modify intrinsics accordingly
        # todo：首尾帧包含12张图像，作为新视角图像
        output_imgs, output_depths, output_depths_m, output_confs_m, output_cks = \
                    load_conditions(output_img_paths, self.reso) # todo output_imgs:(12,3,h,w)
        output_fxs, output_fys, output_cxs, output_cys = output_cks[:, 0, 0], output_cks[:, 1, 1], output_cks[:, 0, 2], output_cks[:, 1, 2]

        # compute image fovs and pixel directions
        output_fovxs, output_fovys = [], []
        for fx, fy, cx, cy in zip(output_fxs, output_fys, output_cxs, output_cys):
            fovx = 2 * np.arctan(cx / fx)
            fovy = 2 * np.arctan(cy / fy)
            output_fovxs.append(fovx)
            output_fovys.append(fovy)
        output_fovxs = torch.as_tensor(output_fovxs, dtype=torch.float32)
        output_fovys = torch.as_tensor(output_fovys, dtype=torch.float32)

        # add input data to output
        output_imgs = torch.cat([output_imgs, input_imgs], dim=0)
        output_depths = torch.cat([output_depths, input_depths], dim=0)
        output_depths_m = torch.cat([output_depths_m, input_depths_m], dim=0)
        output_confs_m = torch.cat([output_confs_m, input_confs_m], dim=0)
        output_c2ws = torch.cat([output_c2ws, input_c2ws], dim=0)
        output_fovxs = torch.cat([output_fovxs, input_fovxs], dim=0)
        output_fovys = torch.cat([output_fovys, input_fovys], dim=0)
        output_fxs = torch.cat([output_fxs, input_fxs], dim=0)
        output_fys = torch.cat([output_fys, input_fys], dim=0)
        output_cxs = torch.cat([output_cxs, input_cxs], dim=0)
        output_cys = torch.cat([output_cys, input_cys], dim=0)
        output_directions = []
        for fx, fy, cx, cy in zip(output_fxs, output_fys, output_cxs, output_cys):
            fovx = 2 * np.arctan(cx / fx)
            fovy = 2 * np.arctan(cy / fy)
            direction = get_ray_directions(self.reso[0], self.reso[1],
                                           focal=[fx, fy], principal=[cx, cy])
            output_directions.append(direction)
        output_directions = torch.stack(output_directions)
        output_rays_o, output_rays_d = get_rays(
                    output_directions, output_c2ws, keepdim=True, normalize=False)
        # todo -----------------------------#
        # todo 将输出内容按re10k数据集整理
        input_h,input_w = input_imgs.shape[-2:]
        input_cks_norm = input_cks.clone()
        input_cks_norm[:, 0, [0, 2]] /= input_w  # fx, cx
        input_cks_norm[:, 1, [1, 2]] /= input_h  # fy, cy

        output_h,output_w = output_imgs.shape[-2:]
        output_cks_norm = torch.stack([
            torch.stack([output_fxs, torch.zeros_like(output_fxs), output_cxs], dim=1),
            torch.stack([torch.zeros_like(output_fys), output_fys, output_cys], dim=1),
            torch.stack([torch.zeros_like(output_fxs), torch.zeros_like(output_fys), torch.ones_like(output_fxs)], dim=1),
        ], dim=1)
        output_cks_norm[:, 0, [0, 2]] /= output_w  # fx, cx
        output_cks_norm[:, 1, [1, 2]] /= output_h  # fy, cy

        example = {
            "context": {
                "extrinsics": input_c2ws,
                "intrinsics": input_cks_norm,
                "image": input_imgs,
                "near": self.get_bound("near", len(input_imgs)),
                "far": self.get_bound("far", len(input_imgs)),
                "index": torch.arange(input_imgs.shape[0])

            },
            "target": {
                "extrinsics": output_c2ws,
                "intrinsics": output_cks_norm,
                "image": output_imgs,
                "near": self.get_bound("near", len(output_imgs)),
                "far": self.get_bound("far", len(output_imgs)),
                "index": torch.arange(output_imgs.shape[0])
            },
            "scene": bin_token,
        }
        return example

        # todo Omni-Scene中格式
        # # pack data
        # input_dict = {"rgb": input_imgs}

        # input_dict_pix = {"depth_m": input_depths_m, "conf_m": input_confs_m,
        #                   "ck": input_cks, "c2w": input_c2ws,
        #                   "cx": input_cxs, "cy": input_cys, "fx": input_fxs, "fy": input_fys,
        #                   "rays_o": input_rays_o, "rays_d": input_rays_d}

        # input_dict_vol = {"w2i": input_w2is}

        # output_dict = {"rgb": output_imgs, "depth": output_depths,
        #                "depth_m": output_depths_m, "conf_m": output_confs_m,
        #                "c2w": output_c2ws, "fovx": output_fovxs, "fovy": output_fovys,
        #                "rays_o": output_rays_o, "rays_d": output_rays_d}

        # return {
        #     "bin_token": bin_token,
        #     "outputs": output_dict, #
        #     "inputs": input_dict,
        #     "inputs_pix": input_dict_pix,
        #     "inputs_vol": input_dict_vol
        # }
