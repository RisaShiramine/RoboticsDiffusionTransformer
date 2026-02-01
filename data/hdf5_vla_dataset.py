import os
import fnmatch
import json
import h5py
import yaml
import cv2
import numpy as np
from configs.state_vec import STATE_VEC_IDX_MAPPING

class HDF5VLADataset:
    """
    适配 RoboTwin 格式的 HDF5 数据集加载器
    """
    def __init__(self) -> None:
        # [Modify] 指向刚才链接的数据集路径
        HDF5_DIR = "data/datasets/robotwin/"
        self.DATASET_NAME = "robotwin"
        
        self.file_paths = []
        # 递归查找 .hdf5 文件
        for root, _, files in os.walk(HDF5_DIR):
            for filename in fnmatch.filter(files, '*.hdf5'):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)
                
        # 加载配置
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISTORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
        
        # Load embedding mapping
        self.embed_mapping = {}
        mapping_path = "data/embeddings/robotwin/mapping.json"
        if os.path.exists(mapping_path):
            print(f"Loading embedding mapping from {mapping_path}")
            with open(mapping_path, 'r') as f:
                self.embed_mapping = json.load(f)
    
        # 计算每个 episode 的长度用于采样权重
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        
        # 简单的归一化权重
        valid_lens = [l for l in episode_lens if l > 0]
        if not valid_lens:
            print("Warning: No valid episodes found!")
            self.episode_sample_weights = []
        else:
            self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
    def __len__(self):
        return len(self.file_paths)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            
            valid, sample = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            
            if valid:
                return sample
            else:
                # 如果当前 index 无效，随机换一个
                index = np.random.randint(0, len(self.file_paths))
    
    def parse_hdf5_file_state_only(self, file_path):
        """只读取状态，用于快速计算长度和统计信息"""
        try:
            with h5py.File(file_path, 'r') as f:
                # RobotTwin: joint_action is used as state
                if 'joint_action' not in f:
                    return False, None
                
                # 1. 加载关节数据
                # RoboTwin: left_arm (6), left_gripper (1), right_arm (6), right_gripper (1)
                left_arm = f['joint_action']['left_arm'][:]
                left_gripper = f['joint_action']['left_gripper'][:].reshape(-1, 1)
                right_arm = f['joint_action']['right_arm'][:]
                right_gripper = f['joint_action']['right_gripper'][:].reshape(-1, 1)
                
                # 拼接成 (N, 14) 的向量 [left_arm, left_gripper, right_arm, right_gripper]
                qpos = np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=1)
                num_steps = qpos.shape[0]

                if num_steps < self.CHUNK_SIZE + 1: # 太短的数据丢弃
                    return False, None
                    
                # 辅助函数：将 14维数据填充到 128维的统一向量中
                def fill_in_state(values):
                    # 根据 configs/state_vec.py 的映射
                    # Values shape: (..., 14)
                    # 0-5: left arm joints -> indices [50-55]
                    # 6: left gripper -> index [60]
                    # 7-12: right arm joints -> indices [0-5]
                    # 13: right gripper -> index [10]
                    
                    uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                    
                    # Left Arm (6 joints)
                    for i in range(6):
                        uni_vec[..., 50+i] = values[..., 0+i] # left_arm is index 0-5
                    # Left Gripper
                    uni_vec[..., 60] = values[..., 6] # left_gripper is index 6
                    
                    # Right Arm (6 joints)
                    for i in range(6):
                        uni_vec[..., 0+i] = values[..., 7+i] # right_arm is index 7-12
                    # Right Gripper
                    uni_vec[..., 10] = values[..., 13] # right_gripper is index 13
                    
                    return uni_vec

                state = fill_in_state(qpos)
                return True, {"state": state}
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return False, None

    def parse_hdf5_file(self, file_path):
        try:
            with h5py.File(file_path, 'r') as f:
                # 1. 加载关节数据 (使用 joint_action 作为状态和动作)
                # RoboTwin: left_arm (6), left_gripper (1), right_arm (6), right_gripper (1)
                left_arm = f['joint_action']['left_arm'][:]
                left_gripper = f['joint_action']['left_gripper'][:].reshape(-1, 1)
                right_arm = f['joint_action']['right_arm'][:]
                right_gripper = f['joint_action']['right_gripper'][:].reshape(-1, 1)
                
                # 拼接成 (N, 14) 的向量 [left_arm, left_gripper, right_arm, right_gripper]
                qpos = np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=1)
                num_steps = qpos.shape[0]
                if num_steps < self.CHUNK_SIZE + 1:
                    return False, None
                
                # 2. 随机采样一个时间步 t
                # 确保后面有足够的长度截取 chunk
                step_id = np.random.randint(0, num_steps - self.CHUNK_SIZE)
                
                # 3. 加载指令 Instruction
                # 路劲映射: data/.../data/episodeX.hdf5 -> data/.../instructions/episodeX.json
                # 获取文件名 episodeX
                file_name = os.path.basename(file_path).replace('.hdf5', '')
                # 获取 dataset 根目录 (即 robotwin 链接的目录)
                # 假设链接后结构是 data/datasets/robotwin/episodeX.hdf5
                # 但实际上您的源文件是在 .../demo_clean/data/ 下, instruction 在 .../demo_clean/instructions/
                # 由于我们做了软链接，os.path.dirname(file_path) 是 data/datasets/robotwin
                # 我们需要找到真实的 instructions 路径，或者假设 instructions 也被拷贝/链接了
                # 最简单的方法是：假设 instructions 就在 data 目录的同级目录
                
                # 这里为了稳健，我们通过 file_path 推断 instruction 路径
                # 如果 file_path 是 link，os.readlink 可以找到源，但这里我们在 python 逻辑里
                # 建议您也将 instructions 文件夹链接到 data/datasets/robotwin_instructions
                # 或者直接用绝对路径硬编码（如果只是临时测试）
                
                # 尝试这套逻辑：
                # HDF5_DIR = "data/datasets/robotwin/" -> /mnt/hdd/RoboTwin/data/stack_blocks_three/demo_clean/data
                # so ../instructions is /mnt/hdd/RoboTwin/data/stack_blocks_three/demo_clean/instructions
                
                instr_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(file_path)), "../instructions"))
                
                # 如果找不到，尝试去原始路径找 (Hardcoded for your setup)
                if not os.path.exists(instr_dir):
                    instr_dir = "/mnt/hdd/RoboTwin/data/stack_blocks_three/demo_clean/instructions"
                
                instr_path = os.path.join(instr_dir, f"{file_name}.json")
                
                instruction = "Do the task." # 默认值
                if os.path.exists(instr_path):
                    with open(instr_path, 'r') as fp:
                        instr_data = json.load(fp)
                        # 在 seen 和 unseen 列表中随机选一句
                        all_instr = instr_data.get('seen', []) + instr_data.get('unseen', [])
                        if all_instr:
                            instruction = np.random.choice(all_instr)
                
                # Use precomputed embedding path if available
                # [Hack] Mimic old project behavior: use hardcoded cached embeddings
                # This bypasses the need for T5-XXL in memory and mapping files
                import random
                lang_embed_idx = random.randint(0, 5)
                # Ensure the path is absolute or relative to where script is run
                # We copied them to data/embeddings/robotwin/
                instruction = os.path.abspath(f"data/embeddings/robotwin/lang_embed_{lang_embed_idx}.pt")
                
                # 4. 组装 Meta 信息
                meta = {
                    "dataset_name": self.DATASET_NAME,
                    "#steps": num_steps,
                    "step_id": step_id,
                    "instruction": instruction
                }

                # 5. 构建 State 和 Action
                # 当前时刻的状态
                state_raw = qpos[step_id:step_id+1] # (1, 14)
                # 未来动作块
                action_raw = qpos[step_id:step_id+self.CHUNK_SIZE] # (CHUNK_SIZE, 14)
                
                # 计算统计量 (基于整个 episode)
                state_std_raw = np.std(qpos, axis=0)
                state_mean_raw = np.mean(qpos, axis=0)
                state_norm_raw = np.sqrt(np.mean(qpos**2, axis=0))

                # 辅助函数：将 14维数据填充到 128维的统一向量中
                def fill_in_state(values):
                    # 根据 configs/state_vec.py 的映射
                    # Values shape: (..., 14)
                    # 0-5: left arm joints -> indices [50-55]
                    # 6: left gripper -> index [60]
                    # 7-12: right arm joints -> indices [0-5]
                    # 13: right gripper -> index [10]
                    
                    uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                    
                    # Left Arm (6 joints)
                    for i in range(6):
                        uni_vec[..., 50+i] = values[..., 0+i] # left_arm is index 0-5 in qpos
                    # Left Gripper
                    uni_vec[..., 60] = values[..., 6] # left_gripper is index 6
                    
                    # Right Arm (6 joints)
                    for i in range(6):
                        uni_vec[..., 0+i] = values[..., 7+i] # right_arm is index 7-12
                    # Right Gripper
                    uni_vec[..., 10] = values[..., 13] # right_gripper is index 13
                    
                    return uni_vec

                state = fill_in_state(state_raw)
                actions = fill_in_state(action_raw)
                state_std = fill_in_state(state_std_raw)
                state_mean = fill_in_state(state_mean_raw)
                state_norm = fill_in_state(state_norm_raw)
                state_indicator = fill_in_state(np.ones_like(state_mean_raw))

                # 6. 读取图像
                # 格式: (IMG_HISTORY_SIZE, H, W, 3)
                # RoboTwin 只有当前时刻图片，没有历史 buffer？
                # 如果没有历史，我们需要自己处理 padding，或者只读当前帧
                # 这里假设我们只取当前帧 t，如果需要历史可以往前读
                
                def load_image(key_name):
                    # key_name 例如 'observation/front_camera/rgb'
                    if key_name not in f:
                         return np.zeros((self.IMG_HISTORY_SIZE, 0, 0, 0), dtype=np.uint8), \
                               np.zeros((self.IMG_HISTORY_SIZE,), dtype=bool)

                    # 数据是 bytes (|S...)，需要 cv2 解码
                    img_bytes = f[key_name][step_id]
                    # np.frombuffer 配合 imdecode
                    # 注意：HDF5 存的可能是 bytes string，直接用 np.void 或 bytes 转换
                    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    if img is None:
                        return np.zeros((self.IMG_HISTORY_SIZE, 0, 0, 0), dtype=np.uint8), \
                               np.zeros((self.IMG_HISTORY_SIZE,), dtype=bool)
                    
                    # 转 RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # 处理历史帧 (这里简单复制当前帧作为历史，或者只填最后一帧)
                    # 标准做法是填入 img_history
                    out_imgs = np.zeros((self.IMG_HISTORY_SIZE, *img.shape), dtype=np.uint8)
                    out_mask = np.zeros((self.IMG_HISTORY_SIZE,), dtype=bool)
                    
                    # 只要当前帧
                    out_imgs[-1] = img
                    out_mask[-1] = True
                    return out_imgs, out_mask

                # 映射摄像头
                cam_high, cam_high_mask = load_image('observation/front_camera/rgb')
                # 假设 left_camera 是左手，right_camera 是右手
                cam_left, cam_left_mask = load_image('observation/left_camera/rgb') 
                cam_right, cam_right_mask = load_image('observation/right_camera/rgb')

                sample = {
                    "meta": meta,
                    "step_id": step_id,
                    "state": state,
                    "state_std": state_std,
                    "state_mean": state_mean,
                    "state_norm": state_norm,
                    "actions": actions,
                    "state_indicator": state_indicator,
                    "cam_high": cam_high,
                    "cam_high_mask": cam_high_mask,
                    "cam_left_wrist": cam_left,
                    "cam_left_wrist_mask": cam_left_mask,
                    "cam_right_wrist": cam_right,
                    "cam_right_wrist_mask": cam_right_mask
                }
                return True, sample

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return False, None
