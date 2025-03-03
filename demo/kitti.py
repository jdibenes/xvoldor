
import numpy as np

class pose_loader:
    def __init__(self, fname):
        self._mat = np.loadtxt(fname, delimiter=' ')

    def get_pose(self, index):
        return np.vstack((self._mat[index, :].reshape((3, 4)), np.array([0,0,0,1]).reshape((1, 4))))
    
    def count(self):
        return self._mat.shape[0]

def trajectoryDistances(poses):
    dist = [0]
    for i in range(1, poses.count()):
        P1 = poses.get_pose(i-1)
        P2 = poses.get_pose(i)
        dx = P1[0,3] - P2[0,3]
        dy = P1[1,3] - P2[1,3]
        dz = P1[2,3] - P2[2,3]
        dist.append(dist[i-1]+np.sqrt(dx*dx+dy*dy+dz*dz))
    return dist

def lastFrameFromSegmentLength(dist, first_frame, leng):
    for i in range(first_frame, len(dist)):
        if (dist[i]>dist[first_frame]+leng):
            return i
    return -1

def rotationError(pose_error):
    a = pose_error[0,0]
    b = pose_error[1,1]
    c = pose_error[2,2]
    d = 0.5*(a+b+c-1.0)
    return np.arccos(max(min(d, 1.0),-1.0))

def translationError(pose_error):
    dx = pose_error[0,3]
    dy = pose_error[1,3]
    dz = pose_error[2,3]
    return np.sqrt(dx*dx+dy*dy+dz*dz)

def calcSequenceErrors(poses_gt, poses_result):
    lengths = [100,200,300,400,500,600,700,800]
    num_lengths = 8
    err = []
    step_size = 10
    dist = trajectoryDistances(poses_gt)
    
    first_frame = 0
    while (first_frame < poses_gt.count()):
        for i in range(0, num_lengths):
            leng = lengths[i]
            last_frame = lastFrameFromSegmentLength(dist, first_frame, leng)
            
            if (last_frame==-1):
                continue

            pose_delta_gt = np.linalg.inv(poses_gt.get_pose(first_frame)) @ poses_gt.get_pose(last_frame)
            pose_delta_result = np.linalg.inv(poses_result.get_pose(first_frame)) @ poses_result.get_pose(last_frame)
            pose_error = np.linalg.inv(pose_delta_result) @ pose_delta_gt
            r_err = rotationError(pose_error)
            t_err = translationError(pose_error)

            num_frames = last_frame - first_frame + 1
            speed = leng / (0.1*num_frames)

            err.append((first_frame, r_err/leng, t_err/leng, leng, speed))
        first_frame += step_size
    return err
