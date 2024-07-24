import numpy as np

def save_ply(fname, points, faces=None, colors=None):
    if faces is None and colors is None:
        points = points.reshape(-1,3)
        to_save = points
        return np.savetxt(fname,
                  to_save,
                  fmt='%.6f %.6f %.6f',
                  comments='',
                  header=(
                      'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nend_header'.format(points.shape[0]))
                        )
    elif faces is None and not colors is None:
        points = points.reshape(-1,3)
        colors = colors.reshape(-1,3)
        to_save = np.concatenate([points, colors], axis=-1)
        return np.savetxt(fname,
                          to_save,
                          fmt='%.6f %.6f %.6f %d %d %d',
                          comments='',
                          header=(
                      'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header'.format(
                      points.shape[0])))

    elif not faces is None and colors is None:
        points = points.reshape(-1,3)
        faces = faces.reshape(-1,3)
        with open(fname,'w') as f:
            f.write('ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nelement face {:d}\nproperty list uchar int vertex_indices\nend_header\n'.format(
                      points.shape[0],faces.shape[0]))
            for i in range(points.shape[0]):
                f.write('%.6f %.6f %.6f\n'%(points[i,0],points[i,1],points[i,2]))
            for i in range(faces.shape[0]):
                f.write('3 %d %d %d\n'%(faces[i,0],faces[i,1],faces[i,2]))
    elif not faces is None and not colors is None:
        points = points.reshape(-1,3)
        colors = colors.reshape(-1,3)
        faces = faces.reshape(-1,3)
        with open(fname,'w') as f:
            f.write('ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {:d}\nproperty list uchar int vertex_indices\nend_header\n'.format(
                      points.shape[0],faces.shape[0]))
            for i in range(points.shape[0]):
                f.write('%.6f %.6f %.6f %d %d %d\n'%(points[i,0],points[i,1],points[i,2],colors[i,0],colors[i,1],colors[i,2]))
            for i in range(faces.shape[0]):
                f.write('3 %d %d %d\n'%(faces[i,0],faces[i,1],faces[i,2]))


def load_ply(mesh_file):
    vertex_data = []
    face_data = []
    vertNum = -1
    faceNum = -1
    startIdx = -1
    with open(mesh_file,'r') as f:
        data = f.readlines()
        for idx in range(20):
            if data[idx][:14] == 'element vertex':
                vertNum = int(data[idx].split(' ')[-1])
            if data[idx][:12] == 'element face':
                faceNum = int(data[idx].split(' ')[-1])
            if data[idx] == 'end_header\n':
                startIdx = idx
        for idx in range(startIdx+1, startIdx + vertNum + 1):
            values = data[idx].split()
            v = list(map(float, values[0:3]))
            vertex_data.append(v)
        for idx in range(startIdx + vertNum + 1, len(data)):
            values = data[idx].split()
            v = list(map(int, values[1:4]))
            face_data.append(v)
    vertices = np.array(vertex_data)
    faces = np.array(face_data)
    return vertices, faces