import numpy as np
import scipy.stats as stats
from ik import MediapipeSkeleton
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d


KEYPOINTS_DICT = {
    'nose': 0,
    'left_inner_eye': 1,
    'left_eye': 2,
    'left_outer_eye': 3,
    'right_inner_eye': 4,
    'right_eye': 5,
    'right_outer_eye': 6,
    'left_ear': 7,
    'right_ear':8,
    'left_mouth': 9,
    'right_mouth': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_outer_hand': 17,
    'right_outer_hand': 18,
    'left_hand_tip': 19,
    'right_hand_tip': 20,
    'left_inner_hand': 21,
    'right_inner_hand': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_toe': 31,
    'right_toe': 32
}

SKELETON_INFO = {
    0:['left_shoulder', 'left_elbow'],
    1:['left_elbow', 'left_wrist'],
    2:['left_wrist', 'left_hand_tip'],
    3:['right_shoulder', 'right_elbow'],
    4:['right_elbow', 'right_wrist'],
    5:['right_wrist', 'right_hand_tip'],
    6:['left_hip', 'left_knee'],
    7:['left_knee', 'left_ankle'],
    8:['left_ankle', 'left_toe'],
    9:['right_hip', 'right_knee'],
    10:['right_knee', 'right_ankle'],
    11:['right_ankle', 'right_toe'],
}


class FeatureExtractor18:
    
    bones = {
        'hips': 0,
        'chest': 1,
        'neck': 2,
        'head': 3,
        'left_upperarm': 4,
        'left_lowerarm': 5,
        'left_hand': 6,
        'right_upperarm': 7,
        'right_lowerarm': 8,
        'right_hand': 9,
        'left_upperleg': 10,
        'left_lowerleg': 11,
        'left_foot': 12,
        'left_toe': 13,
        'right_upperleg': 14,
        'right_lowerleg': 15,
        'right_foot': 16,
        'right_toe': 17
    }
    
    def __init__(self, config):
        self.config = config
        self.use_position = config.get('use_position', True)
        self.use_rotation = config.get('use_rotation', True)
        self.use_velocity = config.get('use_velocity', True)
        self.use_quaternion_diffs = config.get('use_quaternion_diffs', True)
        self.use_angular_velocity = config.get('use_angular_velocity', False)
        self.use_acceleration = config.get('use_acceleration', False)
        self.use_angular_acceleration = config.get('use_angular_acceleration', False)
        self.use_body = config.get('use_body', False)
        self.normalize = config.get('normalize', False)
        self.use_root = config.get('use_root', True)
        self.output_type = config.get('output_type', 'default')
        
    def apply(self, positions, rotations):
        n_frames, n_joints, _ = positions.shape
        positions = positions.copy()
        rotations = rotations.copy()
        features = []

        # 体格情報
        if self.use_body:
            body_shapes = self.calculate_body_shape(positions)
            features.append(np.tile(body_shapes, (n_frames, 1)))

        # 正規化
        if self.normalize:
            y_min = positions[:,:,1].min()
            positions[:,:,1] = positions[:,:,1] - y_min
            y_max = positions[:,3,1].mean()  # 3 -> Head
            scale_factor = 1 / y_max  
            positions = positions * scale_factor

        if self.use_position or self.use_velocity or self.use_acceleration:
            position = positions.copy()
            # 相対化
            if self.use_root:
                position[:,1:] -= position[:,:1]
            else:
                position -= position[:,:1]
                
            if self.use_position:
                features.append(position)

        if self.use_rotation:        
            rotation = rotations.copy()            
            if self.use_rotation:
                features.append(rotation)

        if self.use_velocity or self.use_acceleration:
            velocity = differential(positions) 
            if self.use_velocity:
                features.append(velocity)

        if self.use_quaternion_diffs:
            quaternion_diffs = calculate_quaternion_difference(rotations)
            features.append(quaternion_diffs)

        if self.use_angular_velocity or self.use_angular_acceleration:
            angular_velocity = calculate_angular_velocity(rotations)
            if self.use_angular_velocity:
                features.append(angular_velocity)

        if self.use_acceleration:
            acceleration = differential(velocity) 
            features.append(acceleration)

        if self.use_angular_acceleration:
            angular_acceleration = differential(angular_velocity)
            features.append(angular_acceleration)

        features = np.concatenate(features, axis=-1)
        
        if self.output_type == "graph":
            return features
        else:
            return features.reshape([len(features),-1])

    def calculate_body_shape(self, positions):
        neck = positions[:,self.bones['neck']]
        head = positions[:,self.bones['head']]
        hip = positions[:,self.bones['hips']]
        body_length = stats.norm.fit(np.linalg.norm(neck - hip, axis=-1))[0]
        head = stats.norm.fit(np.linalg.norm(head - neck, axis=-1))[0] * 2

        l_upperarm_length = calculate_bone_length(positions, self.bones['left_upperarm'], self.bones['left_lowerarm'])
        l_lowerarm_length = calculate_bone_length(positions, self.bones['left_lowerarm'], self.bones['left_hand'])
        r_upperarm_length = calculate_bone_length(positions, self.bones['right_upperarm'], self.bones['right_lowerarm'])
        r_lowerarm_length = calculate_bone_length(positions, self.bones['right_lowerarm'], self.bones['right_hand'])

        l_upperleg_length = calculate_bone_length(positions, self.bones['left_upperleg'], self.bones['left_lowerleg'])
        l_lowerleg_length = calculate_bone_length(positions, self.bones['left_lowerleg'], self.bones['left_foot'])
        r_upperleg_length = calculate_bone_length(positions, self.bones['right_upperleg'], self.bones['right_lowerleg'])
        r_lowerleg_length = calculate_bone_length(positions, self.bones['right_lowerleg'], self.bones['right_foot'])

        arm_length = (l_upperarm_length + r_upperarm_length) / 2 + (l_lowerarm_length + r_lowerarm_length) / 2
        leg_length = (l_upperleg_length + r_upperleg_length) / 2 + (l_lowerleg_length + r_lowerleg_length) / 2
        height = leg_length + body_length + head
        return [height]


# calculate relative positions from root
def align_center(keypoints3d):
    center = (keypoints3d[KEYPOINTS_DICT['left_hip']] + keypoints3d[KEYPOINTS_DICT['right_hip']]) / 2
    return keypoints3d - center[:,None]


# calculate vectors from skeleton
def get_vectors(keypoints3d, normalize=False):
    root = (keypoints3d[KEYPOINTS_DICT['left_hip']] +  keypoints3d[KEYPOINTS_DICT['right_hip']]) / 2
    neck = (keypoints3d[KEYPOINTS_DICT['left_shoulder']] +  keypoints3d[KEYPOINTS_DICT['right_shoulder']]) / 2
    head = (keypoints3d[KEYPOINTS_DICT['left_ear']] +  keypoints3d[KEYPOINTS_DICT['right_ear']]) / 2
    vecs_dict = {
        'neck': neck - root,
        'head': head - neck,
        
        'left_shoulder': keypoints3d[KEYPOINTS_DICT['left_shoulder']] - neck,
        'left_upper_arm': keypoints3d[KEYPOINTS_DICT['left_elbow']] - keypoints3d[KEYPOINTS_DICT['left_shoulder']],
        'left_lower_arm': keypoints3d[KEYPOINTS_DICT['left_wrist']] - keypoints3d[KEYPOINTS_DICT['left_elbow']],
        'left_hand': keypoints3d[KEYPOINTS_DICT['left_hand_tip']] - keypoints3d[KEYPOINTS_DICT['left_wrist']],
        
        'right_shoulder': keypoints3d[KEYPOINTS_DICT['right_shoulder']] - neck,
        'right_upper_arm': keypoints3d[KEYPOINTS_DICT['right_elbow']] - keypoints3d[KEYPOINTS_DICT['right_shoulder']],
        'right_lower_arm': keypoints3d[KEYPOINTS_DICT['right_wrist']] - keypoints3d[KEYPOINTS_DICT['right_elbow']],
        'right_hand': keypoints3d[KEYPOINTS_DICT['right_hand_tip']] - keypoints3d[KEYPOINTS_DICT['right_wrist']],
        
        'left_hip': keypoints3d[KEYPOINTS_DICT['left_hip']] - root,
        'left_upper_leg': keypoints3d[KEYPOINTS_DICT['left_knee']] - keypoints3d[KEYPOINTS_DICT['left_hip']],
        'left_lower_leg': keypoints3d[KEYPOINTS_DICT['left_ankle']] - keypoints3d[KEYPOINTS_DICT['left_knee']],
        'left_foot': keypoints3d[KEYPOINTS_DICT['left_toe']] - keypoints3d[KEYPOINTS_DICT['left_ankle']],
    
        'right_hip': keypoints3d[KEYPOINTS_DICT['right_hip']] - root,
        'right_upper_leg': keypoints3d[KEYPOINTS_DICT['right_knee']] - keypoints3d[KEYPOINTS_DICT['right_hip']],
        'right_lower_leg': keypoints3d[KEYPOINTS_DICT['right_ankle']] - keypoints3d[KEYPOINTS_DICT['right_knee']],
        'right_foot': keypoints3d[KEYPOINTS_DICT['right_toe']] - keypoints3d[KEYPOINTS_DICT['right_ankle']],
    }
    vectors = []
    for vec in vecs_dict.values():
        if normalize:
            if np.linalg.norm(vec) > 0:
                vec = vec / np.linalg.norm(vec)
            else:
                vec = np.zeros(vec.shape)
        vectors.append(vec)
    vectors = np.array(vectors)
    return vectors


# calculate differential
def differential(array):
    array_pad = np.concatenate([array[:1], array, array[-1:]])
    diff = array_pad[2:] - array_pad[:-2]
    return diff
    

def calculate_quaternion_difference(rotations):
    n_frames, n_joints, _ = rotations.shape

    # 連続するフレーム間のクォータニオンを取得
    rotations_pad = np.concatenate([rotations[:1], rotations], axis=0)
    rotations_current = rotations_pad[:-1].reshape(-1, 4)
    rotations_next = rotations_pad[1:].reshape(-1, 4)

    # 差分クォータニオンの計算
    diffs = (R.from_quat(rotations_next) * R.from_quat(rotations_current).inv()).as_quat()

    return diffs.reshape(n_frames, n_joints, 4)
    

def calculate_angular_velocity(rotations, delta_time=1/60):
    n_frames, n_joints, _ = rotations.shape

    # クォータニオンをオイラー角に変換
    euler_angles = R.from_quat(rotations.reshape(-1, 4)).as_euler('xyz').reshape(n_frames, n_joints, 3)

    # 連続するフレーム間のオイラー角の差を計算    
    euler_angles_pad = np.concatenate([euler_angles[:1], euler_angles], axis=0)
    euler_diff = np.diff(euler_angles_pad, axis=0)

    # 角速度の計算
    angular_velocity = euler_diff / delta_time

    return angular_velocity


def normalize_keypoints3d(keypoints3d_list):
    # shape => (n_frames, n_joints, 3)
    # normalization
    y_min = keypoints3d_list[:,:,1].min()
    keypoints3d_list[:,:,1] = keypoints3d_list[:,:,1] - y_min
    y_max = keypoints3d_list[:,KEYPOINTS_DICT['nose'],1].max()
    scale_factor = 1 / y_max  
    keypoints3d_list = keypoints3d_list * scale_factor
    return keypoints3d_list


def data_augmentation(keypoints3d_list, alpha=0.0, is_mirror=False, scaling=False):
    # shape => (n_frames, n_joints, 3)
    
    # spatial scaling
    if scaling:
        scale_factor = np.random.uniform(0.9, 1.1)
        keypoints3d_list *= scale_factor
    
    # gaussian noise
    if alpha > 0:
        std  = np.std(keypoints3d_list, axis=0)
        keypoints3d_list = keypoints3d_list + alpha * np.random.normal(loc=0.0, scale=std, size=keypoints3d_list.shape)
        
    # left-right flip
    if is_mirror:
        keypoints3d_list[:,:,0] = -keypoints3d_list[:,:,0]
        
    return keypoints3d_list


def spatial_augmentation(positions, rotations=None, alpha=0.0, flip=False, scaling=False):
    # shape => (n_frames, n_joints, 3)
    positions = positions.copy()
    if rotations is not None:
        rotations = rotations.copy()
    
    # spatial scaling
    if scaling:
        scale_factor = np.random.uniform(0.8, 1.2)
        positions = positions * scale_factor
    
    # gaussian noise
    if alpha > 0:
        std  = np.std(positions, axis=0)
        positions = positions + alpha * np.random.normal(loc=0.0, scale=std, size=positions.shape)
        if rotations is not None:
            std  = np.std(rotations, axis=0)
            rotations = rotations + alpha * np.random.normal(loc=0.0, scale=std, size=rotations.shape)
        
    # left-right flip
    if flip:
        positions[:,:,0] = -positions[:,:,0] # pos_x
        if rotations is not None:
            rotations[:,:,1] = -rotations[:,:,1] # rot_y
            rotations[:,:,2] = -rotations[:,:,2] # rot_z
        
    return positions, rotations


def interpolate(data, n_frames):
    length = len(data)
    t = np.arange(length)
    x = np.linspace(0, length - 1, n_frames)
    y = interp1d(t, data, axis=0, fill_value="extrapolate")(x)
    return y

def align_with_crop_or_pad(data, n_frames):
    if n_frames > len(data):
        # パディング
        pad_length = n_frames - len(data)
        pad = np.zeros((pad_length,) + data.shape[1:])
        return np.concatenate((data, pad), axis=0)
    else:
        # スライス
        return data[:n_frames]    

def temporal_augmentation(data, method='uniform', temporal_scaling=False):
    # 時系列スケーリング
    scaling_factor = np.random.uniform(0.7, 1.5) if temporal_scaling else 1.0
        
    length = len(data)
    n_frames = int(length * scaling_factor)
    data = interpolate(data, n_frames * 4)

    if method == 'uniform':
        indices = np.linspace(0, len(data) - 1, n_frames, dtype=int)
    elif method == 'random':
        indices = np.sort(np.random.choice(len(data), n_frames, replace=False))
    elif method == 'speedup':
        indices = np.cumsum(np.arange(n_frames))
        indices = indices * (len(data) - 1) // indices[-1]
    elif method == 'slowdown':
        indices = np.cumsum(np.arange(n_frames))
        indices = indices * (len(data) - 1) // indices[-1]
        indices = indices[-1] - indices
    else:
        raise ValueError("Unknown augmentation method: {}".format(method))

    sampled_data = data[np.sort(indices)]
    return align_with_crop_or_pad(sampled_data, length)


def extract_features(
    keypoints3d_list,
    rotations, 
    use_root = True,
    use_position = True,
    use_rotation = True,
    use_velocity = True,
    use_quaternion_diffs = False,
    use_angular_velocity = False,
    use_acceleration = False,
    use_angular_acceleration = False,
    use_quaternion = False,
    use_body = False,
    normalize = False):
    
    n_frames, n_joints, _ = keypoints3d_list.shape
    features = []

    # 体格情報
    if use_body:
        body_shapes = extract_body_shape(keypoints3d_list)
        features.append(np.tile(body_shapes, (n_frames, 1)))
    
    # グローバルな情報
    root_positions = (keypoints3d_list[:,KEYPOINTS_DICT['left_hip']] + keypoints3d_list[:,KEYPOINTS_DICT['right_hip']]) / 2
    root_velocity = differential(root_positions)    
    if use_root:
        features.append(root_positions)
        features.append(root_velocity)
    
    # センタリング
    keypoints3d_list = keypoints3d_list - root_positions[:,None]

    # 正規化
    if normalize:
        keypoints3d_list = normalize_keypoints3d(keypoints3d_list)
    
    if use_position or use_velocity or use_acceleration:
        position = keypoints3d_list.reshape([n_frames, -1])
        if use_position:
            features.append(position)
    
    if use_rotation:        
        if use_quaternion:             
            rotation = rotations.reshape([n_frames, -1])
        else:
            rotation = []
            for keypoints3d in keypoints3d_list:
                vectors = get_vectors(keypoints3d, normalize=normalize)
                rotation.append(vectors)
            rotation = np.array(rotation).reshape([n_frames, -1])
            
        if use_rotation:
            features.append(rotation)
        
    if use_velocity or use_acceleration:
        velocity = differential(position) 
        if use_velocity:
            features.append(velocity)

    if use_quaternion_diffs:
        quaternion_diffs = calculate_quaternion_difference(rotations)
        features.append(quaternion_diffs.reshape([n_frames, -1]))
        
    if use_angular_velocity or use_angular_acceleration:
        angular_velocity = calculate_angular_velocity(rotations).reshape([n_frames, -1])
        if use_angular_velocity:
            features.append(angular_velocity)
        
    if use_acceleration:
        acceleration = differential(velocity) 
        features.append(acceleration)
        
    if use_angular_acceleration:
        angular_acceleration = differential(angular_velocity)
        features.append(angular_acceleration)
        
    features = np.concatenate(features, axis=-1)
    return features



class LMA:
    
    features_all = ['weight', 'space', 'time', 'shape']
    
    @staticmethod
    def calculate_displacement(motion_data):
        displacement = np.diff(motion_data, axis=0)
        displacement = np.insert(displacement, 0, 0, axis=0)  # Restore original shape
        return displacement

    @staticmethod
    def calculate_weight(motion_data, ar=1.0, af=1.0, at=1.0):
        displacement = LMA.calculate_displacement(motion_data)
        weight_per_marker = np.sum(displacement ** 2, axis=-1)
        total_weight = ar * np.mean(weight_per_marker[:,[KEYPOINTS_DICT['left_hip'], KEYPOINTS_DICT['right_hip']]], axis=1) 
        total_weight += af * np.mean(weight_per_marker[:,[KEYPOINTS_DICT['left_wrist'], KEYPOINTS_DICT['right_wrist'], KEYPOINTS_DICT['left_hand_tip'], KEYPOINTS_DICT['right_hand_tip']]], axis=1)
        total_weight += at * np.mean(weight_per_marker[:,[KEYPOINTS_DICT['left_toe'], KEYPOINTS_DICT['right_toe']]], axis=1)
        return total_weight

    @staticmethod
    def calculate_normal_vector(motion_data):
        v1 = motion_data[:,KEYPOINTS_DICT['nose'],:] - motion_data[:,KEYPOINTS_DICT['right_shoulder'],:]
        v2 = motion_data[:,KEYPOINTS_DICT['left_shoulder'],:] - motion_data[:,KEYPOINTS_DICT['right_shoulder'],:]
        normal_vector = np.cross(v1, v2)
        return normal_vector

    @staticmethod
    def calculate_space(motion_data):
        root = (motion_data[:,KEYPOINTS_DICT['left_hip'],:] + motion_data[:,KEYPOINTS_DICT['right_hip'],:]) / 2.0
        displacement = np.diff(root, axis=0)
        displacement = np.insert(displacement, 0, 0, axis=0)  # Restore original shape
        normal_vector = LMA.calculate_normal_vector(motion_data)
        return np.sum(normal_vector * displacement, axis=-1)

    @staticmethod
    def calculate_velocity(motion_data):
        displacement = LMA.calculate_displacement(motion_data)
        velocity = np.sqrt(np.sum(displacement ** 2, axis=-1))
        return velocity

    @staticmethod
    def calculate_acceleration(motion_data):
        velocity = LMA.calculate_velocity(motion_data)
        acceleration = np.diff(velocity, axis=0)
        acceleration = np.insert(acceleration, 0, 0, axis=0)  # Restore original shape
        return acceleration

    @staticmethod
    def calculate_time(motion_data):
        acceleration = LMA.calculate_acceleration(motion_data)
        return np.sum(acceleration, axis=1)

    @staticmethod
    def calculate_bounding_box(motion_data):
        min_coords = np.min(motion_data, axis=1)
        max_coords = np.max(motion_data, axis=1)
        return max_coords - min_coords

    @staticmethod
    def calculate_shape(motion_data):
        bounding_box = LMA.calculate_bounding_box(motion_data)
        shape_change = np.diff(bounding_box, axis=0)
        shape_change = np.insert(shape_change, 0, 0, axis=0)  # Restore original shape
        return shape_change



def rotation_matrix_from_vectors(forward_vector, up_vector):
    
    if np.allclose(forward_vector, 0) or np.allclose(up_vector, 0):
        # Return the identity matrix if either vector is zero
        return np.identity(3)
        
    # Normalize the input vectors
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    up_vector = up_vector / np.linalg.norm(up_vector)
    
    # Compute the right vector using cross product
    right_vector = np.cross(forward_vector, up_vector)
    right_vector /= np.linalg.norm(right_vector)
    
    # Compute the actual up vector using cross product
    forward_vector = -np.cross(right_vector, up_vector)
    
    # Construct the rotation matrix
    R = np.column_stack((-right_vector, up_vector, forward_vector))
    
    return R
    

def matrix_to_quaternion(matrix):
    # Ensure the matrix is a numpy array
    matrix = np.asarray(matrix)
    
    # Calculate the trace of the matrix
    tr = matrix.trace()
    
    # Check the trace value to perform the suitable calculation
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (matrix[2, 1] - matrix[1, 2]) / S
        y = (matrix[0, 2] - matrix[2, 0]) / S
        z = (matrix[1, 0] - matrix[0, 1]) / S
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        S = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
        w = (matrix[2, 1] - matrix[1, 2]) / S
        x = 0.25 * S
        y = (matrix[0, 1] + matrix[1, 0]) / S
        z = (matrix[0, 2] + matrix[2, 0]) / S
    elif matrix[1, 1] > matrix[2, 2]:
        S = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
        w = (matrix[0, 2] - matrix[2, 0]) / S
        x = (matrix[0, 1] + matrix[1, 0]) / S
        y = 0.25 * S
        z = (matrix[1, 2] + matrix[2, 1]) / S
    else:
        S = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
        w = (matrix[1, 0] - matrix[0, 1]) / S
        x = (matrix[0, 2] + matrix[2, 0]) / S
        y = (matrix[1, 2] + matrix[2, 1]) / S
        z = 0.25 * S
        
    return x, y, z, w


def quaternion_to_matrix(quaternion):
    x, y, z, w = quaternion
    matrix = np.zeros([3,3])
    matrix[0,0] = 2*w**2 + 2*x**2 - 1
    matrix[0,1] = 2*x*y - 2*z*w
    matrix[0,2] = 2*x*z + 2*y*w
    matrix[1,0] = 2*x*y + 2*z*w
    matrix[1,1] = 2*w**2 + 2*y**2 - 1
    matrix[1,2] = 2*y*z - 2*x*w
    matrix[2,0] = 2*x*z - 2*y*w
    matrix[2,1] = 2*y*z + 2*x*w
    matrix[2,2] = 2*w**2 + 2*z**2 - 1
    return matrix
    

# 関節の回転を計算するための関数
def calculate_body_rotations(pose):
    nose_pos = pose[KEYPOINTS_DICT['nose']]
    l_ear_pos = pose[KEYPOINTS_DICT['left_ear']]
    r_ear_pos = pose[KEYPOINTS_DICT['right_ear']]
    l_eye_pos = pose[KEYPOINTS_DICT['left_eye']]
    r_eye_pos = pose[KEYPOINTS_DICT['right_eye']]
    l_mouth_pos = pose[KEYPOINTS_DICT['left_mouth']]    
    r_mouth_pos = pose[KEYPOINTS_DICT['right_mouth']]
    
    l_shoulder_pos = pose[KEYPOINTS_DICT['left_shoulder']]
    l_elbow_pos = pose[KEYPOINTS_DICT['left_elbow']]
    l_wrist_pos = pose[KEYPOINTS_DICT['left_wrist']]
    l_inner_hand_pos = pose[KEYPOINTS_DICT['left_inner_hand']]
    l_outer_hand_pos = pose[KEYPOINTS_DICT['left_outer_hand']]   
    l_hand_tip_pos = pose[KEYPOINTS_DICT['left_hand_tip']]
    
    r_shoulder_pos = pose[KEYPOINTS_DICT['right_shoulder']]
    r_elbow_pos = pose[KEYPOINTS_DICT['right_elbow']]
    r_wrist_pos = pose[KEYPOINTS_DICT['right_wrist']]
    r_inner_hand_pos = pose[KEYPOINTS_DICT['right_inner_hand']]
    r_outer_hand_pos = pose[KEYPOINTS_DICT['right_outer_hand']]
    r_hand_tip_pos = pose[KEYPOINTS_DICT['right_hand_tip']]

    l_hip_pos = pose[KEYPOINTS_DICT['left_hip']]
    l_knee_pos = pose[KEYPOINTS_DICT['left_knee']]
    l_ankle_pos = pose[KEYPOINTS_DICT['left_ankle']]
    l_heel_pos = pose[KEYPOINTS_DICT['left_heel']]
    l_toe_pos = pose[KEYPOINTS_DICT['left_toe']]
    
    r_hip_pos = pose[KEYPOINTS_DICT['right_hip']]
    r_knee_pos = pose[KEYPOINTS_DICT['right_knee']]
    r_ankle_pos = pose[KEYPOINTS_DICT['right_ankle']]
    r_heel_pos = pose[KEYPOINTS_DICT['right_heel']]
    r_toe_pos = pose[KEYPOINTS_DICT['right_toe']]
    
    hip_pos = (l_hip_pos + r_hip_pos) / 2
    mouth_pos = (l_mouth_pos + r_mouth_pos) / 2
    head_pos = (l_ear_pos + r_ear_pos + mouth_pos) / 3 
    neck_pos = (l_shoulder_pos + r_shoulder_pos) / 2
    
    rotations = {}

    # 腰    
    hip_forward = np.cross(r_hip_pos - neck_pos, l_hip_pos - neck_pos)
    hip_right = r_hip_pos - l_hip_pos
    hip_up = np.cross(hip_right, hip_forward)
    hip_rotation_matrix = rotation_matrix_from_vectors(hip_forward, hip_up)
    rotations['hips'] = matrix_to_quaternion(hip_rotation_matrix)

    # 上半身    
    body_forward = np.cross(l_shoulder_pos - hip_pos, r_shoulder_pos - hip_pos)
    body_right = r_shoulder_pos - l_shoulder_pos
    body_up = np.cross(body_right, body_forward)
    body_rotation_matrix = rotation_matrix_from_vectors(body_forward, body_up)
    rotations['body'] = matrix_to_quaternion(body_rotation_matrix)

    # 首の回転
    neck_up = (head_pos - neck_pos)
    neck_forward = body_forward  # 上半身の前方向を使用
    neck_rotation_matrix = rotation_matrix_from_vectors(neck_forward, neck_up)
    rotations['neck'] = matrix_to_quaternion(neck_rotation_matrix)

    # 頭の回転
    head_forward = np.cross(l_eye_pos - mouth_pos, r_eye_pos - mouth_pos)
    head_right = r_ear_pos - l_ear_pos
    head_up = np.cross(head_right, head_forward)
    head_rotation_matrix = rotation_matrix_from_vectors(head_forward, head_up)
    rotations['head'] = matrix_to_quaternion(head_rotation_matrix)

    # ... 左腕の回転を計算 ...    
    # 左上腕の回転
    l_upperarm_right = -(l_elbow_pos - l_shoulder_pos)
    l_upperarm_up = np.cross(l_shoulder_pos - l_elbow_pos, l_wrist_pos - l_elbow_pos)
    l_upperarm_forward = np.cross(l_upperarm_up, l_upperarm_right)
    l_upperarm_up = np.cross(l_upperarm_right, l_upperarm_forward)
    l_upperarm_rotation_matrix = rotation_matrix_from_vectors(l_upperarm_forward, l_upperarm_up)
    rotations['left_upperarm'] = matrix_to_quaternion(l_upperarm_rotation_matrix)

    # 左下腕の回転    
    l_lowerarm_right = -(l_wrist_pos - l_elbow_pos)
    l_lowerarm_forward = l_inner_hand_pos - l_outer_hand_pos
    l_lowerarm_up = np.cross(l_lowerarm_right, l_lowerarm_forward)
    l_lowerarm_forward = np.cross(l_lowerarm_up, l_lowerarm_right)
    l_lowerarm_rotation_matrix = rotation_matrix_from_vectors(l_lowerarm_forward, l_lowerarm_up)
    rotations['left_lowerarm'] = matrix_to_quaternion(l_lowerarm_rotation_matrix)
    
    # 左手の回転    
    l_hand_right = -(l_hand_tip_pos - l_wrist_pos)
    l_hand_up = np.cross(l_inner_hand_pos - l_wrist_pos, l_outer_hand_pos - l_wrist_pos)    
    l_hand_forward = np.cross(l_hand_up, l_hand_right)
    l_hand_up = np.cross(l_hand_right, l_hand_forward)
    l_hand_rotation_matrix = rotation_matrix_from_vectors(l_hand_forward, l_hand_up)
    rotations['left_hand'] = matrix_to_quaternion(l_hand_rotation_matrix)

     # 右上腕の回転
    r_upperarm_right = (r_elbow_pos - r_shoulder_pos)
    r_upperarm_up = np.cross(r_wrist_pos - r_elbow_pos, r_shoulder_pos - r_elbow_pos)
    r_upperarm_forward = np.cross(r_upperarm_up, r_upperarm_right)
    r_upperarm_up = np.cross(r_upperarm_right, r_upperarm_forward)
    r_upperarm_rotation_matrix = rotation_matrix_from_vectors(r_upperarm_forward, r_upperarm_up)
    rotations['right_upperarm'] = matrix_to_quaternion(r_upperarm_rotation_matrix)

    # 右下腕の回転
    r_lowerarm_right = (r_wrist_pos - r_elbow_pos)  
    r_lowerarm_forward = r_inner_hand_pos - r_outer_hand_pos
    r_lowerarm_up = np.cross(r_lowerarm_right, r_lowerarm_forward)
    r_lowerarm_forward = np.cross(r_lowerarm_up, r_lowerarm_right)
    r_lowerarm_rotation_matrix = rotation_matrix_from_vectors(r_lowerarm_forward, r_lowerarm_up)
    rotations['right_lowerarm'] = matrix_to_quaternion(r_lowerarm_rotation_matrix)

    # 右手の回転
    r_hand_right = (r_hand_tip_pos - r_wrist_pos) 
    r_hand_up = np.cross(r_outer_hand_pos - r_wrist_pos, r_inner_hand_pos - r_wrist_pos)
    r_hand_forward = np.cross(r_hand_up, r_hand_right)
    r_hand_up = np.cross(r_hand_right, r_hand_forward)
    r_hand_rotation_matrix = rotation_matrix_from_vectors(r_hand_forward, r_hand_up)
    rotations['right_hand'] = matrix_to_quaternion(r_hand_rotation_matrix)

    # 左上脚の回転
    l_upperleg_up = -(l_knee_pos - l_hip_pos)
    l_knee_angle = np.degrees(np.arccos(np.dot((l_hip_pos - l_knee_pos), (l_ankle_pos - l_knee_pos)) / 
                        (np.linalg.norm(l_hip_pos - l_knee_pos) * np.linalg.norm(l_ankle_pos - l_knee_pos))))
    if (l_knee_angle < 160):
        l_upperleg_right = np.cross(l_hip_pos - l_knee_pos, l_ankle_pos - l_knee_pos)
    else:
        l_upperleg_right = np.cross(l_heel_pos - l_hip_pos, l_toe_pos - l_hip_pos)
    l_upperleg_forward = np.cross(l_upperleg_up, l_upperleg_right)
    l_upperleg_rotation_matrix = rotation_matrix_from_vectors(l_upperleg_forward, l_upperleg_up)
    rotations['left_upperleg'] = matrix_to_quaternion(l_upperleg_rotation_matrix)

    # 左膝の回転
    l_lowerleg_up = -(l_ankle_pos - l_knee_pos)
    l_lowerleg_right = np.cross(l_heel_pos - l_knee_pos, l_toe_pos - l_knee_pos)
    l_lowerleg_forward = np.cross(l_lowerleg_up, l_lowerleg_right)
    l_lowerleg_rotation_matrix = rotation_matrix_from_vectors(l_lowerleg_forward, l_lowerleg_up)
    rotations['left_lowerleg'] = matrix_to_quaternion(l_lowerleg_rotation_matrix)

    # 左足首の回転
    l_foot_forward = l_toe_pos - l_heel_pos
    l_foot_right = np.cross(l_heel_pos - l_ankle_pos, l_toe_pos - l_ankle_pos)
    l_foot_up = np.cross(l_foot_right, l_foot_forward)
    l_foot_rotation_matrix = rotation_matrix_from_vectors(l_foot_forward, l_foot_up)
    rotations['left_foot'] = matrix_to_quaternion(l_foot_rotation_matrix)

    # 右上脚の回転
    r_upperleg_up = -(r_knee_pos - r_hip_pos)
    r_knee_angle = np.degrees(np.arccos(np.dot((r_hip_pos - r_knee_pos), (r_ankle_pos - r_knee_pos)) / 
                        (np.linalg.norm(r_hip_pos - r_knee_pos) * np.linalg.norm(r_ankle_pos - r_knee_pos))))
    if (r_knee_angle < 160):
        r_upperleg_right = np.cross(r_hip_pos - r_knee_pos, r_ankle_pos - r_knee_pos)  
    else:
        r_upperleg_right = np.cross(r_heel_pos - r_hip_pos, r_toe_pos - r_hip_pos) 
    r_upperleg_forward = np.cross(r_upperleg_up, r_upperleg_right)
    r_upperleg_rotation_matrix = rotation_matrix_from_vectors(r_upperleg_forward, r_upperleg_up)
    rotations['right_upperleg'] = matrix_to_quaternion(r_upperleg_rotation_matrix)

    # 右膝の回転
    r_lowerleg_up = -(r_ankle_pos - r_knee_pos)
    r_lowerleg_right = np.cross(r_heel_pos - r_knee_pos, r_toe_pos - r_knee_pos)
    r_lowerleg_forward = np.cross(r_lowerleg_up, r_lowerleg_right)
    r_lowerleg_rotation_matrix = rotation_matrix_from_vectors(r_lowerleg_forward, r_lowerleg_up)
    rotations['right_lowerleg'] = matrix_to_quaternion(r_lowerleg_rotation_matrix)
    
    # 右足首の回転
    r_foot_forward = r_toe_pos - r_heel_pos
    r_foot_right = np.cross(r_heel_pos - r_ankle_pos, r_toe_pos - r_ankle_pos)  
    r_foot_up = np.cross(r_foot_right, r_foot_forward)  
    r_foot_rotation_matrix = rotation_matrix_from_vectors(r_foot_forward, r_foot_up)
    rotations['right_foot'] = matrix_to_quaternion(r_foot_rotation_matrix)

    return rotations


def extract_body_shape(keypoints3d_list):
    neck = (keypoints3d_list[:,KEYPOINTS_DICT['left_shoulder']] + keypoints3d_list[:,KEYPOINTS_DICT['right_shoulder']]) / 2
    hip = (keypoints3d_list[:,KEYPOINTS_DICT['left_hip']] + keypoints3d_list[:,KEYPOINTS_DICT['right_hip']]) / 2
    body_length = stats.norm.fit(np.linalg.norm(neck - hip, axis=-1))[0]
    
    head = stats.norm.fit(np.linalg.norm(keypoints3d_list[:,KEYPOINTS_DICT['nose']] - neck, axis=-1))[0] * 2
    
    l_upperarm_length = calculate_bone_length(keypoints3d_list, KEYPOINTS_DICT['left_shoulder'], KEYPOINTS_DICT['left_elbow'])
    l_lowerarm_length = calculate_bone_length(keypoints3d_list, KEYPOINTS_DICT['left_elbow'], KEYPOINTS_DICT['left_wrist'])
    r_upperarm_length = calculate_bone_length(keypoints3d_list, KEYPOINTS_DICT['right_shoulder'], KEYPOINTS_DICT['right_elbow'])
    r_lowerarm_length = calculate_bone_length(keypoints3d_list, KEYPOINTS_DICT['right_elbow'], KEYPOINTS_DICT['right_wrist'])
    
    l_upperleg_length = calculate_bone_length(keypoints3d_list, KEYPOINTS_DICT['left_hip'], KEYPOINTS_DICT['left_knee'])
    l_lowerleg_length = calculate_bone_length(keypoints3d_list, KEYPOINTS_DICT['left_knee'], KEYPOINTS_DICT['left_ankle'])
    r_upperleg_length = calculate_bone_length(keypoints3d_list, KEYPOINTS_DICT['right_hip'], KEYPOINTS_DICT['right_knee'])
    r_lowerleg_length = calculate_bone_length(keypoints3d_list, KEYPOINTS_DICT['right_knee'], KEYPOINTS_DICT['right_ankle'])
    
    arm_length = (l_upperarm_length + r_upperarm_length) / 2 + (l_lowerarm_length + r_lowerarm_length) / 2
    leg_length = (l_upperleg_length + r_upperleg_length) / 2 + (l_lowerleg_length + r_lowerleg_length) / 2
    height = leg_length + body_length + head
    return [height]


def calculate_bone_length(keypoints3d, joint_index_1, joint_index_2):
    distances = np.linalg.norm(keypoints3d[:, joint_index_1] - keypoints3d[:, joint_index_2], axis=-1)
    mu, std = stats.norm.fit(distances)
    return mu

