import numpy as np
import scipy.ndimage.filters as filters


class MediapipeSkeleton:

    KEYPOINT_DICT = {
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
        'right_toe': 32,
        # extra joint for kinematic chain
        'root': 33,
        'neck': 34,
    }

    KINEMATIC_TREE = [
        [KEYPOINT_DICT['root'], KEYPOINT_DICT['left_hip']],
        [KEYPOINT_DICT['root'], KEYPOINT_DICT['right_hip']],
        [KEYPOINT_DICT['root'], KEYPOINT_DICT['neck']],
        [KEYPOINT_DICT['neck'], KEYPOINT_DICT['left_shoulder'], KEYPOINT_DICT['left_elbow'], KEYPOINT_DICT['left_wrist'], KEYPOINT_DICT['left_hand_tip']],
        [KEYPOINT_DICT['neck'], KEYPOINT_DICT['right_shoulder'], KEYPOINT_DICT['right_elbow'], KEYPOINT_DICT['right_wrist'], KEYPOINT_DICT['right_hand_tip']],
        [KEYPOINT_DICT['left_hip'], KEYPOINT_DICT['left_knee'], KEYPOINT_DICT['left_ankle'], KEYPOINT_DICT['left_heel'], KEYPOINT_DICT['left_toe']],
        [KEYPOINT_DICT['right_hip'], KEYPOINT_DICT['right_knee'], KEYPOINT_DICT['right_ankle'], KEYPOINT_DICT['right_heel'], KEYPOINT_DICT['right_toe']],
        [KEYPOINT_DICT['neck'], KEYPOINT_DICT['nose']],
        [KEYPOINT_DICT['nose'], KEYPOINT_DICT['left_eye']],
        [KEYPOINT_DICT['nose'], KEYPOINT_DICT['right_eye']],
        [KEYPOINT_DICT['nose'], KEYPOINT_DICT['left_ear']],
        [KEYPOINT_DICT['nose'], KEYPOINT_DICT['right_ear']],
        [KEYPOINT_DICT['nose'], KEYPOINT_DICT['left_mouth']],
        [KEYPOINT_DICT['nose'], KEYPOINT_DICT['right_mouth']]
    ]

    def __init__(self):
        self.offset = None
        self.initialized = False

    def inverse_kinematics_np(self, pose, smooth_forward=False):
        # pose must be shape (batch_size, joints=33, 3)
        if not self.initialized:
            print('skeleton dnot initialized')
            return None
        
        root_pos = (pose[:,self.KEYPOINT_DICT['left_hip']] + pose[:,self.KEYPOINT_DICT['right_hip']]) / 2
        neck_pos = (pose[:,self.KEYPOINT_DICT['left_shoulder']] + pose[:,self.KEYPOINT_DICT['right_shoulder']]) / 2
        joints = np.concatenate([pose, root_pos[:,np.newaxis,:], neck_pos[:,np.newaxis,:]], axis=1)  # Extended pose including 'root' and 'neck'
        
        '''Get Forward Direction'''
        l_hip, r_hip, sdr_r, sdr_l = self.KEYPOINT_DICT['left_hip'], self.KEYPOINT_DICT['right_hip'], self.KEYPOINT_DICT['left_shoulder'], self.KEYPOINT_DICT['right_shoulder']
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]
        # print(across1.shape, across2.shape)

        # forward (batch_size, 3)
        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        if smooth_forward:
            forward = filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')
            # forward (batch_size, 3)
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

        '''Get Root Rotation'''
        target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)

        '''Inverse Kinematics'''
        # quat_params (batch_size, joints_num, 4)
        # print(joints.shape[:-1])
        quat_params = np.zeros(joints.shape[:-1] + (4,))
        # print(quat_params.shape)
        root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        quat_params[:, 0] = root_quat
        # quat_params[0, 0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        for chain in self.KINEMATIC_TREE:
            R = root_quat
            for j in range(len(chain) - 1):
                # (batch, 3)
                u = self.offset[chain[j+1]][np.newaxis,...].repeat(len(joints), axis=0)
                # print(u.shape)
                # (batch, 3)
                v = joints[:, chain[j+1]] - joints[:, chain[j]]
                v = v / np.sqrt((v**2).sum(axis=-1))[:, np.newaxis]
                # print(u.shape, v.shape)
                rot_u_v = qbetween_np(u, v)

                R_loc = qmul_np(qinv_np(R), rot_u_v)

                quat_params[:,chain[j + 1], :] = R_loc
                R = qmul_np(R, R_loc)

        quat_params = np.nan_to_num(quat_params, 0)
        return quat_params


    def initialize(self, pose):
        # Calculate joint lengths and tpose offsets
        KEYPOINT_DICT = self.KEYPOINT_DICT
        kinematic_tree = self.KINEMATIC_TREE

        joint_lengths = {}
        tpose_offsets = np.zeros((len(self.KEYPOINT_DICT), 3))

        # Calculate root and neck position
        root_pos = (pose[KEYPOINT_DICT['left_hip']] + pose[KEYPOINT_DICT['right_hip']]) / 2
        neck_pos = (pose[KEYPOINT_DICT['left_shoulder']] + pose[KEYPOINT_DICT['right_shoulder']]) / 2

        extended_pose = np.concatenate([pose, root_pos[np.newaxis,:], neck_pos[np.newaxis,:]], axis=0)  # Extended pose including 'root' and 'neck'

        for chain in kinematic_tree:
            for j in range(len(chain) - 1):
                joint_lengths[(chain[j], chain[j+1])] = np.linalg.norm(extended_pose[chain[j+1]] - extended_pose[chain[j]])

        # Compute T-Pose offsets based on joint lengths.
        tpose_offsets[KEYPOINT_DICT['root']] = [0, 0, 0]  # root is center

        # Upper body
        tpose_offsets[KEYPOINT_DICT['neck']] = [0, joint_lengths[(KEYPOINT_DICT['root'], KEYPOINT_DICT['neck'])], 0]
        tpose_offsets[KEYPOINT_DICT['left_shoulder']] = [-joint_lengths[(KEYPOINT_DICT['neck'], KEYPOINT_DICT['left_shoulder'])], joint_lengths[(KEYPOINT_DICT['root'], KEYPOINT_DICT['neck'])], 0]
        tpose_offsets[KEYPOINT_DICT['right_shoulder']] = [joint_lengths[(KEYPOINT_DICT['neck'], KEYPOINT_DICT['right_shoulder'])], joint_lengths[(KEYPOINT_DICT['root'], KEYPOINT_DICT['neck'])], 0]

        # Arm segments
        tpose_offsets[KEYPOINT_DICT['left_elbow']] = tpose_offsets[KEYPOINT_DICT['left_shoulder']] + [-joint_lengths[(KEYPOINT_DICT['left_shoulder'], KEYPOINT_DICT['left_elbow'])], 0, 0]
        tpose_offsets[KEYPOINT_DICT['right_elbow']] = tpose_offsets[KEYPOINT_DICT['right_shoulder']] + [joint_lengths[(KEYPOINT_DICT['right_shoulder'], KEYPOINT_DICT['right_elbow'])], 0, 0]

        tpose_offsets[KEYPOINT_DICT['left_wrist']] = tpose_offsets[KEYPOINT_DICT['left_elbow']] + [-joint_lengths[(KEYPOINT_DICT['left_elbow'], KEYPOINT_DICT['left_wrist'])], 0, 0]
        tpose_offsets[KEYPOINT_DICT['right_wrist']] = tpose_offsets[KEYPOINT_DICT['right_elbow']] + [joint_lengths[(KEYPOINT_DICT['right_elbow'], KEYPOINT_DICT['right_wrist'])], 0, 0]
        
        tpose_offsets[KEYPOINT_DICT['left_hand_tip']] = tpose_offsets[KEYPOINT_DICT['left_wrist']] + [-joint_lengths[(KEYPOINT_DICT['left_wrist'], KEYPOINT_DICT['left_hand_tip'])], 0, 0]
        tpose_offsets[KEYPOINT_DICT['right_hand_tip']] = tpose_offsets[KEYPOINT_DICT['right_wrist']] + [joint_lengths[(KEYPOINT_DICT['right_wrist'], KEYPOINT_DICT['right_hand_tip'])], 0, 0]

        # Lower body
        tpose_offsets[KEYPOINT_DICT['left_hip']] = [-joint_lengths[(KEYPOINT_DICT['root'], KEYPOINT_DICT['left_hip'])], 0, 0]
        tpose_offsets[KEYPOINT_DICT['right_hip']] = [joint_lengths[(KEYPOINT_DICT['root'], KEYPOINT_DICT['right_hip'])], 0, 0]

        # Leg segments
        tpose_offsets[KEYPOINT_DICT['left_knee']] = tpose_offsets[KEYPOINT_DICT['left_hip']] + [0, -joint_lengths[(KEYPOINT_DICT['left_hip'], KEYPOINT_DICT['left_knee'])], 0]
        tpose_offsets[KEYPOINT_DICT['right_knee']] = tpose_offsets[KEYPOINT_DICT['right_hip']] + [0, -joint_lengths[(KEYPOINT_DICT['right_hip'], KEYPOINT_DICT['right_knee'])], 0]

        tpose_offsets[KEYPOINT_DICT['left_ankle']] = tpose_offsets[KEYPOINT_DICT['left_knee']] + [0, -joint_lengths[(KEYPOINT_DICT['left_knee'], KEYPOINT_DICT['left_ankle'])], 0]
        tpose_offsets[KEYPOINT_DICT['right_ankle']] = tpose_offsets[KEYPOINT_DICT['right_knee']] + [0, -joint_lengths[(KEYPOINT_DICT['right_knee'], KEYPOINT_DICT['right_ankle'])], 0]

        tpose_offsets[KEYPOINT_DICT['left_heel']] = tpose_offsets[KEYPOINT_DICT['left_ankle']] + [0, -joint_lengths[(KEYPOINT_DICT['left_ankle'], KEYPOINT_DICT['left_heel'])], 0]
        tpose_offsets[KEYPOINT_DICT['right_heel']] = tpose_offsets[KEYPOINT_DICT['right_ankle']] + [0, -joint_lengths[(KEYPOINT_DICT['right_ankle'], KEYPOINT_DICT['right_heel'])], 0]
        
        tpose_offsets[KEYPOINT_DICT['left_toe']] = tpose_offsets[KEYPOINT_DICT['left_heel']] + [0, 0, joint_lengths[(KEYPOINT_DICT['left_heel'], KEYPOINT_DICT['left_toe'])]]
        tpose_offsets[KEYPOINT_DICT['right_toe']] = tpose_offsets[KEYPOINT_DICT['right_heel']] + [0, 0, joint_lengths[(KEYPOINT_DICT['right_heel'], KEYPOINT_DICT['right_toe'])]]

        # Facial features
        tpose_offsets[KEYPOINT_DICT['nose']] = tpose_offsets[KEYPOINT_DICT['neck']] + [0, joint_lengths[(KEYPOINT_DICT['neck'], KEYPOINT_DICT['nose'])], 0]
        tpose_offsets[KEYPOINT_DICT['left_eye']] = tpose_offsets[KEYPOINT_DICT['nose']] + [-np.sqrt(joint_lengths[(KEYPOINT_DICT['nose'], KEYPOINT_DICT['left_eye'])]), np.sqrt(joint_lengths[(KEYPOINT_DICT['nose'], KEYPOINT_DICT['left_eye'])]), 0]
        tpose_offsets[KEYPOINT_DICT['right_eye']] = tpose_offsets[KEYPOINT_DICT['nose']] + [np.sqrt(joint_lengths[(KEYPOINT_DICT['nose'], KEYPOINT_DICT['right_eye'])]), np.sqrt(joint_lengths[(KEYPOINT_DICT['nose'], KEYPOINT_DICT['right_eye'])]), 0]
        tpose_offsets[KEYPOINT_DICT['left_ear']] = tpose_offsets[KEYPOINT_DICT['nose']] + [-joint_lengths[(KEYPOINT_DICT['nose'], KEYPOINT_DICT['left_ear'])], 0, 0]
        tpose_offsets[KEYPOINT_DICT['right_ear']] = tpose_offsets[KEYPOINT_DICT['nose']] + [joint_lengths[(KEYPOINT_DICT['nose'], KEYPOINT_DICT['right_ear'])], 0, 0]
        tpose_offsets[KEYPOINT_DICT['left_mouth']] = tpose_offsets[KEYPOINT_DICT['nose']] + [-np.sqrt(joint_lengths[(KEYPOINT_DICT['nose'], KEYPOINT_DICT['left_mouth'])]), -np.sqrt(joint_lengths[(KEYPOINT_DICT['nose'], KEYPOINT_DICT['left_mouth'])]), 0]
        tpose_offsets[KEYPOINT_DICT['right_mouth']] = tpose_offsets[KEYPOINT_DICT['nose']] + [np.sqrt(joint_lengths[(KEYPOINT_DICT['nose'], KEYPOINT_DICT['right_mouth'])]), -np.sqrt(joint_lengths[(KEYPOINT_DICT['nose'], KEYPOINT_DICT['right_mouth'])]), 0]

        self.offset = tpose_offsets
        self.initialized = True

        return joint_lengths, tpose_offsets



def qbetween_np(v0, v1):
    '''
    find the quaternion used to rotate v0 to v1
    '''
    assert v0.shape[-1] == 3, 'v0 must be of the shape (*, 3)'
    assert v1.shape[-1] == 3, 'v1 must be of the shape (*, 3)'

    v = np.cross(v0, v1)
    w = np.sqrt((v0 ** 2).sum(axis=-1, keepdims=True) * (v1 ** 2).sum(axis=-1, keepdims=True)) + (v0 * v1).sum(axis=-1, keepdims=True)
    return qnormalize(np.concatenate([w, v], axis=-1))


def qnormalize(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def qinv_np(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = np.ones(q.shape)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qmul_np(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized arrays of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a array of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = np.einsum('...i,...j->...ij', r, q)

    w = terms[..., 0, 0] - terms[..., 1, 1] - terms[..., 2, 2] - terms[..., 3, 3]
    x = terms[..., 0, 1] + terms[..., 1, 0] + terms[..., 3, 2] - terms[..., 2, 3]
    y = terms[..., 0, 2] - terms[..., 1, 3] + terms[..., 2, 0] + terms[..., 3, 1]
    z = terms[..., 0, 3] + terms[..., 1, 2] - terms[..., 2, 1] + terms[..., 3, 0]
    
    return np.stack((w, x, y, z), axis=-1).reshape(original_shape)
