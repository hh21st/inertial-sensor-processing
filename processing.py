import numpy as np
import quaternion
import visualization as vis
from reader import OrebaReader

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
        self.quat = np.quaternion(1,0,0,0)

    def set_nodes(self, nodes):
        self.nodes = nodes

    def set_faces(self, faces):
        self.faces = faces

    # TODO here, the quaterion accumulates the rotations instead of actually rotating the nodes
    def rotate_quaternion(self, w, dt):
        self.quat = dt/2 * self.quat * np.quaternion(0, w[0], w[1], w[2]) + self.quat

    def rotate_point(self, point):
        return quaternion.rotate_vectors(self.quat, point)

    def convert_to_computer_frame(self, point):
        computerFrameChangeMatrix = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
        return np.matmul(computerFrameChangeMatrix, point)

    def get_euler_attitude(self):
        def _rad2deg(rad):
            return rad / np.pi * 180
        m = quaternion.as_rotation_matrix(self.quat)
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
    loopRate = 64
    gyro = reader.read_gyro()
    pv = vis.PygameViewer(640, 480, cuboid, 16)
    running = True
    i = 0
    for ang_rate_x, ang_rate_y, ang_rate_z in gyro:
        cuboid.rotate_quaternion([ang_rate_x, ang_rate_y, ang_rate_z], 1/loopRate)
        #print(cuboid.quat)
        if i % 4 == 0:
            if not pv.update():
                break
        i += 1

if __name__ == '__main__':
    process()
