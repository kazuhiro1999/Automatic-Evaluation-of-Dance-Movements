import numpy as np

from preprocess.ik import MediapipeSkeleton


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


def data_augmentation(keypoints3d_list, alpha=0.0, is_mirror=False):
    # shape => (n_frames, n_joints, 3)
    
    # gaussian noise
    scale  = np.std(keypoints3d_list, axis=0)
    if alpha > 0:
        keypoints3d_list = keypoints3d_list + alpha * np.random.normal(loc=0.0, scale=scale, size=keypoints3d_list.shape)
        
    # left-right flip
    if is_mirror:
        keypoints3d_list[:,:,0] = -keypoints3d_list[:,:,0]
        
    return keypoints3d_list


def extract_features(
    keypoints3d_list,
    use_root = True,
    use_position = True,
    use_rotation = True,
    use_velocity = True,
    use_angler_velocity = False,
    use_acceleration = False,
    use_angler_acceleration = False,
    use_quaternion = False,
    normalize = False):
    
    n_frames, n_joints, _ = keypoints3d_list.shape
    features = []
    
    # グローバルな情報
    root_positions = (keypoints3d_list[:,KEYPOINTS_DICT['left_hip']] + keypoints3d_list[:,KEYPOINTS_DICT['right_hip']]) / 2
    root_velocity = differential(root_positions)    
    if use_root:
        features.append(root_positions)
        features.append(root_velocity)
    
    # センタリング
    keypoints3d_list = keypoints3d_list - root_positions[:,None]
    
    if use_position or use_velocity or use_acceleration:
        position = keypoints3d_list.reshape([n_frames, -1])
        if use_position:
            features.append(position)
    
    if use_rotation or use_angler_velocity or use_angler_acceleration:
        if use_quaternion:
            skel = MediapipeSkeleton()
            avg_pose = np.mean(keypoints3d_list, axis=0) # Average pose.
            skel.initialize(avg_pose)
            # apply inverse kinematics
            rotation = skel.inverse_kinematics_np(keypoints3d_list)
            rotation = rotation.reshape([n_frames, -1])
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
        
    if use_angler_velocity or use_angler_acceleration:
        angler_velocity = differential(rotation)
        if use_angler_velocity:
            features.append(angler_velocity)
        
    if use_acceleration:
        acceleration = differential(velocity)
        features.append(acceleration)
        
    if use_angler_acceleration:
        angler_acceleration = differential(angler_velocity)
        features.append(angler_acceleration)
        
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
    
    @staticmethod
    def calculate_features(motion_data):
        weight_features = LMA.calculate_weight(motion_data).reshape([-1,1])
        space_features = LMA.calculate_space(motion_data).reshape([-1,1])
        time_features = LMA.calculate_time(motion_data).reshape([-1,1])
        shape_features = LMA.calculate_shape(motion_data)
        features = np.concatenate([weight_features, space_features, time_features, shape_features], axis=1)
        return features