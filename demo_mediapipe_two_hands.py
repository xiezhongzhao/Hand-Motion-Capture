import cv2
import torch
from manopth import manolayer
from utils import func, bone, AIK, smoother, mediapipe_hand
import numpy as np
from utils.op_pso import PSO
import open3d

class TwoHandsCapture():
    def __init__(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._mano_root = 'mano/models'

        self.pose, self.shape = func.initiate("zero")  # pose.shape: (1,48), shape.shape: (1,10)
        self.pose0 = torch.eye(3).repeat(1, 16, 1, 1)  # shape: (1,16,3,3)

    def mano_init(self, str):
        mano = manolayer.ManoLayer(flat_hand_mean=True,
                                   side=str,
                                   mano_root=self._mano_root,
                                   use_pca=False,
                                   root_rot_mode='rotmat',
                                   joint_rot_mode='rotmat')
        return mano

    def filter_init(self):
        point_filter = smoother.OneEuroFilter(4.0, 0.0)
        shape_filter = smoother.OneEuroFilter(4.0, 0.0)
        mesh_filter = smoother.OneEuroFilter(4.0, 0.0)
        return point_filter, shape_filter, mesh_filter

    def open3d_init(self, mano):
        mesh = open3d.geometry.TriangleMesh()
        hand_verts, j3d_recon = mano(self.pose0, self.shape.float())
        mesh.triangles = open3d.utility.Vector3iVector(mano.th_faces)
        hand_verts = hand_verts.clone().detach().cpu().numpy()[0]
        mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
        return mesh, hand_verts

    def open3d_viewer(self, mesh_right, mesh_left):
        viewer = open3d.visualization.Visualizer()
        viewer.create_window(width=1000, height=480, window_name='mesh')
        viewer.add_geometry(mesh_right)
        viewer.add_geometry(mesh_left)
        viewer.update_renderer()
        return viewer

    def handJoints(self, hand_landmarks):
        pre_joints = list()
        for id, landmark in enumerate(hand_landmarks.landmark):
            pre_joints.append([landmark.x, landmark.y, landmark.z])
        return pre_joints

    def filterHandJoints(self, pre_joints, point_filter):
        pre_joints = np.array(pre_joints)
        pre_joints = point_filter.process(pre_joints)  # filter the jitter joints
        return pre_joints

    def caculateBoneLength(self, pre_joints):
        pre_useful_bone_len = bone.caculate_length(pre_joints, label="useful")
        return pre_useful_bone_len

    def handShapePSO(self, pre_useful_bone_len, shape_filter):
        # get the hand shape with pso algorithm
        NGEN = 0
        popsize = 100
        low = np.zeros((1, 10)) - 3.0
        up = np.zeros((1, 10)) + 3.0
        parameters = [NGEN, popsize, low, up]

        pso = PSO(parameters, pre_useful_bone_len.reshape((1, 15)), self._mano_root)
        pso.main()
        opt_shape = pso.ng_best
        opt_shape = shape_filter.process(opt_shape)
        return opt_shape


    def jointsRotation(self, pre_joints, opt_shape, mano):
        opt_tensor_shape = torch.tensor(opt_shape, dtype=torch.float)
        _, j3d_p0_ops = mano(self.pose0, opt_tensor_shape)  # pose0_shape: (1,16,3,3), opt_shape: (10,) j3d_shape: (1,21,3)
        template = j3d_p0_ops.cpu().numpy().squeeze(0) / 1000.0  # template, m 21*3
        ratio = np.linalg.norm(template[9] - template[0]) / np.linalg.norm(pre_joints[9] - pre_joints[0])
        j3d_pre_process = pre_joints * ratio  # template, m
        j3d_pre_process = j3d_pre_process - j3d_pre_process[0] + template[0]  # shape: (21, 3)
        pose_R = AIK.adaptive_IK(template, j3d_pre_process)  # shape: (1,16,3,3)
        pose_R = torch.from_numpy(pose_R).float()  # shape: (1,16,3,3)
        return pose_R


    def reconstuctHand(self, opt_shape, pose_R, mesh_filter, mesh, mano, offset):
        view_mat = np.array([[1.0, 0.0, 0.0],
                             [0.0, -1.0, 0],
                             [0.0, 0, -1.0]])
        opt_tensor_shape = torch.tensor(opt_shape, dtype=torch.float)
        new_tran = np.array([[0, 0, 0]])
        hand_verts, j3d_recon = mano(pose_R, opt_tensor_shape.float())  # verts.shape:(1,77 8,3), j3d.shape:(1,21,3)
        hand_verts = hand_verts.clone().detach().cpu().numpy()[0]
        hand_verts = mesh_filter.process(hand_verts)
        hand_verts = np.matmul(view_mat, hand_verts.T).T
        hand_verts[:, 0] = hand_verts[:, 0]
        hand_verts[:, 1] = hand_verts[:, 1]
        mesh_tran = np.array([[offset, 50, new_tran[0, 2]]])
        hand_verts = hand_verts - mesh_tran

        mesh.triangles = open3d.utility.Vector3iVector(mano.th_faces)
        mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
        mesh.paint_uniform_color([228 / 255, 178 / 255, 148 / 255])
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

        return mesh

    def jointsMesh(self, pre_joints, point_filter, shape_filter, mesh_filter, mesh, mano, offset):
        pre_joints = np.array(pre_joints)
        pre_joints = self.filterHandJoints(pre_joints, point_filter)

        # calculate bone
        pre_useful_bone_len = self.caculateBoneLength(pre_joints)  # shape: (1,15)
        # pso
        opt_shape = self.handShapePSO(pre_useful_bone_len, shape_filter)
        # change joints into rotation
        pose_R = self.jointsRotation(pre_joints, opt_shape, mano) # shape: (1,16,3,3)
        # reconstruction
        mesh = self.reconstuctHand(opt_shape, pose_R,
                              mesh_filter, mesh, mano, offset)
        return mesh

def main():
    hc = TwoHandsCapture()

    mano_right = hc.mano_init("right")
    mano_left = hc.mano_init("left")

    point_filter_right, shape_filter_right, mesh_filter_right = hc.filter_init()
    point_filter_left, shape_filter_left, mesh_filter_left = hc.filter_init()

    mesh_right, hand_verts_right = hc.open3d_init(mano_right)
    mesh_left, hand_verts_left = hc.open3d_init(mano_left)

    print('start pose estimate')
    viewer = hc.open3d_viewer(mesh_right, mesh_left)
    hands = mediapipe_hand.MediapipeHand()
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        # process the image
        ret_flag, img = cap.read()

        results = hands.get_hands_info(img)
        if results.multi_hand_landmarks:
            for hand_landmarks, handness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # left
                if handness.classification[0].label == "Right":
                    pre_joints_left = hc.handJoints(hand_landmarks)
                    mesh_left = hc.jointsMesh(pre_joints_left,
                                           point_filter_left, shape_filter_left, mesh_filter_left,
                                           mesh_left, mano_left, -200)
                # right
                elif handness.classification[0].label == "Left":
                    pre_joints_right = hc.handJoints(hand_landmarks)
                    mesh_right = hc.jointsMesh(pre_joints_right,
                                           point_filter_right, shape_filter_right, mesh_filter_right,
                                           mesh_right, mano_right, 200)

            viewer.update_geometry(mesh_right)
            viewer.update_geometry(mesh_left)
            viewer.poll_events()

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
























