from typing import Dict, Callable, Tuple
import numpy as np
from diffusion_policy.common.cv2_util import get_image_transform

##此文件用于将real_world环境的相机等obs数据转换为模型所需的obs格式
def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    ''' 
    shape_meta: &shape_meta
  # acceptable types: rgb, low_dim
  obs:
    # camera_0:
    #   shape: ${task.image_shape} =[3, 240, 320]
    #   type: rgb
    camera_1:
      shape: ${task.image_shape}=[3, 240, 320]
      type: rgb
    # camera_2:
    #   shape: ${task.image_shape}=[3, 240, 320]
    #   type: rgb
    camera_3:
      shape: ${task.image_shape}=[3, 240, 320]
      type: rgb
    # camera_4:
    #   shape: ${task.image_shape}=[3, 240, 320]
    #   type: rgb
    robot_eef_pose:
      shape: [2]
      type: low_dim
  action: 
    shape: [2]
    '''
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():#key为相机名称，attr为数据shape和type的字典
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                #使用cv2进行resize
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    # convert to float32 and normalize to [0,1]
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim':
            this_data_in = env_obs[key]
            #this_data_in=（T_o，D=6）
            if 'pose' in key and shape == (2,):
                # take X,Y coordinates
                #this_data_in=（T_o，D=2）
                this_data_in = this_data_in[...,[0,1]]
            obs_dict_np[key] = this_data_in
    return obs_dict_np


def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res
