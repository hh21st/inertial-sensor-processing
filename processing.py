import numpy as np
import quaternion
import visualization as vis
from fusion import MadgwickFusion
from reader import OrebaReader

LOOP_RATE = 64
UPDATE_RATE = 16

# A node is an edge of the cuboid
class Node:
    def __init__(self, coords, color):
        self.x = coords[0]
        self.y = coords[1]
        self.z = coords[2]
        self.color = color

# A face of the cuboid is defined using the indices of four nodes
class Face:
    def __init__(self, nodeIdxs, color):
        self.nodeIdxs = nodeIdxs
        self.color = color

# The cuboid
class Cuboid:
    def __init__(self):
        self.nodes = []
        self.faces = []
        self.q = np.quaternion(1, 0, 0, 0) # Initial pose estimate

    def set_nodes(self, nodes):
        self.nodes = nodes

    def set_faces(self, faces):
        self.faces = faces

    def set_quaternion(self, q):
        self.q = q

    def rotate_quaternion(self, w, dt):
        self.q = dt/2 * self.q * np.quaternion(0, w[0], w[1], w[2]) + self.q

    def rotate_point(self, point):
        return quaternion.rotate_vectors(self.q, point)

    def convert_to_computer_frame(self, point):
        computerFrameChangeMatrix = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
        return np.matmul(computerFrameChangeMatrix, point)

    def get_euler_attitude(self):
        def _rad2deg(rad):
            return rad / np.pi * 180
        m = quaternion.as_rotation_matrix(self.q)
        test = -m[2, 0]
        if test > 0.99999:
            yaw = 0
            pitch = np.pi / 2
            roll = np.arctan2(m[0, 1], m[0, 2])
        elif test < -0.99999:
            yaw = 0
            pitch = -np.pi / 2
            roll = np.arctan2(-m[0, 1], -m[0, 2])
        else:
            yaw = np.arctan2(m[1, 0], m[0, 0])
            pitch = np.arcsin(-m[2, 0])
            roll = np.arctan2(m[2, 1], m[2, 2])
        yaw = _rad2deg(yaw)
        pitch = _rad2deg(pitch)
        roll = _rad2deg(roll)
        return yaw, pitch, roll

def initialize_cuboid():

    # The cuboid
    cuboid = Cuboid()

    # Define nodes
    nodes = []
    nodes.append(Node([-1.5, -1, -0.1], [255, 255, 255]))
    nodes.append(Node([-1.5, -1, 0.1], [255, 255, 255]))
    nodes.append(Node([-1.5, 1, -0.1], [255, 255, 255]))
    nodes.append(Node([-1.5, 1, 0.1], [255, 255, 255]))
    nodes.append(Node([1.5, -1, -0.1], [255, 255, 255]))
    nodes.append(Node([1.5, -1, 0.1], [255, 255, 255]))
    nodes.append(Node([1.5, 1, -0.1], [255, 255, 255]))
    nodes.append(Node([1.5, 1, 0.1], [255, 255, 255]))
    cuboid.set_nodes(nodes)

    # Define faces
    faces = []
    faces.append(Face([0, 2, 6, 4], [255, 0, 255]))
    faces.append(Face([0, 1, 3, 2], [255, 0, 0]))
    faces.append(Face([1, 3, 7, 5], [0, 255, 0]))
    faces.append(Face([4, 5, 7, 6], [0, 0, 255]))
    faces.append(Face([2, 3, 7, 6], [0, 255, 255]))
    faces.append(Face([0, 1, 5, 4], [255, 255, 0]))
    cuboid.set_faces(faces)

    return cuboid

def process():
    cuboid = initialize_cuboid()
    reader = OrebaReader('1004_sensorexport.csv')
    madgwick = MadgwickFusion(cuboid.q, LOOP_RATE)
    acc, gyro = reader.read_inert()
    pv = vis.PygameViewer(640, 480, cuboid, UPDATE_RATE)
    running = True
    i = 0
    for acc_t, gyro_t in zip(acc, gyro):
        # Sensor fusion update
        madgwick.update(acc_t, gyro_t)
        cuboid.set_quaternion(madgwick.q)
        # Update screen every
        if i % (LOOP_RATE//UPDATE_RATE) == 0:
            if not pv.update():
                break
        i += 1

if __name__ == '__main__':
    process()
