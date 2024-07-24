"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2023-04-19
"""
import os
import os.path as osp
import sys
sys.path.append('..')
import random
import time
import pandas
import json
from multiprocessing import Pool
import cv2
import av
from av import time_base as AV_TIME_BASE
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
from tools.omni_tools import makePath
from tools.skel_tools import loadMotion
from tools.cam_tools import loadCamera

def init_params(pos, H, W):
    output = {}

    output["w"] = W
    output["h"] = H
    output["aabb_scale"] = 1
    output["scale"] = 0.46 / 1000.0  # scale to fit person into unit box
    output["offset"] = [-0.6, 0.0, 0.7]  # studio

    x, y, z = pos
    scale_offset = 0.01
    scale = 1.0 / 2.0 * float(y) - scale_offset
    scale = max(0.4, scale)
    offset_x = 0.5 - float(x) * scale
    offset_z = 0.5 - float(z) * scale

    print('scale ' + str(scale))
    print('offset_x ' + str(offset_x))
    print('offset_z ' + str(offset_z))
    output["scale"] = scale / 1000.0
    output["offset"] = [offset_x, 0, offset_z]
    output["from_na"] = True

    return output

def cropImg(img, mask, BGR = True):
    global cropImgSize, imgSize
    CROP_H, CROP_W = cropImgSize
    IMAGE_H, IMAGE_W = imgSize

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 170, 255, cv2.THRESH_BINARY)

    img = np.where(np.stack([mask,mask,mask], -1), img, 0)
    concat_img = np.concatenate([img, mask[:,:,None]], axis=-1)
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

    delta_x = (CROP_W - w) / 2.0
    if delta_x > 0:
        x -= int(delta_x)

    delta_y = (CROP_H - h) / 2.0
    if delta_y > 0:
        y -= int(delta_y)

    w = CROP_W
    h = CROP_H

    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w > IMAGE_W:
        x = IMAGE_W - w
    if y + h > IMAGE_H:
        y = IMAGE_H - h

    CENTER_OFFSET = [y, x]
    cropped_image = concat_img[y:y + h, x:x + w]
    return cropped_image, CENTER_OFFSET

def get_keyframe_interval(cap):
    frame_number = 0
    fps = cap.streams.video[0].average_rate
    video_stream = cap.streams.video[0]
    assert int(1 / video_stream.time_base) % fps == 0
    offset_timestamp = int(1 / video_stream.time_base / fps)
    video_stream.codec_context.skip_frame = "NONKEY"
    target_timestamp = int((frame_number * AV_TIME_BASE) / video_stream.average_rate)
    cap.seek(target_timestamp)
    result = []
    iter = 0
    for frame in cap.decode(video_stream):
        if (iter > 1):
            video_stream.codec_context.skip_frame = "DEFAULT"
            return result[1] - result[0]
            break
        result.append(int(frame.pts / offset_timestamp))
        iter += 1
    video_stream.codec_context.skip_frame = "DEFAULT"
    return -1

def get_timestamp_offset(cap):
    frame_number = 0
    fps = cap.streams.video[0].average_rate
    video_stream = cap.streams.video[0]
    assert int(1 / video_stream.time_base) % fps == 0
    offset_timestamp = int(1 / video_stream.time_base / fps)
    target_timestamp = int((frame_number * AV_TIME_BASE) / video_stream.average_rate)
    cap.seek(target_timestamp)
    for packet in cap.demux():
        for frame in packet.decode():
            return int(frame.dts / offset_timestamp)
    return -1

def get_frame_av(cap, frame_number, index_offset, keyframe_interval):
    fps = cap.streams.video[0].average_rate
    video_stream = cap.streams.video[0]
    assert int(1 / video_stream.time_base) % fps == 0
    offset_timestamp = int(1 / video_stream.time_base / fps)
    video_stream = cap.streams.video[0]

    target_frame = int(frame_number / keyframe_interval) * keyframe_interval
    target_timestamp = int(cap.duration * float(target_frame / int(cap.streams.video[0].frames)))
    cap.seek(target_timestamp)
    framex_index = -1
    for packet in cap.demux():
        for frame in packet.decode():
            if frame.dts:
                framex_index = int(frame.dts / offset_timestamp) - index_offset
            else:
                framex_index += 1
            if (framex_index == frame_number):
                return None
    return None

def cropImgMask(camIdx):
    print("start camIdx: ", camIdx)
    img_video_path = os.path.join(img_video_dir, f"stream{camIdx:03d}.mp4")
    if not os.path.isfile(img_video_path):
        img_video_path = os.path.join(img_video_dir, f"stream{camIdx:03d}.avi")
    imgCap = av.open(img_video_path)

    mask_video_path = os.path.join(mask_video_dir, f"stream{camIdx:03d}.mp4")
    if not os.path.isfile(mask_video_path):
        mask_video_path = os.path.join(mask_video_dir, f"stream{camIdx:03d}.avi")
    maskCap = av.open(mask_video_path)

    print(img_video_path)
    print(mask_video_path)

    # numFrame = imgCap.get(cv2.CAP_PROP_FRAME_COUNT)
    numFrame = imgCap.streams.video[0].frames
    print(numFrame)

    IMAGE_W = imgCap.streams.video[0].format.width
    IMAGE_H = imgCap.streams.video[0].format.height
    print("Height: {} Width: {} from video".format(IMAGE_H, IMAGE_W))

    index_offset = [-1, -1]
    keyframe_interval = [-1, -1]
    keyframe_interval[0] = get_keyframe_interval(imgCap)
    keyframe_interval[1] = get_keyframe_interval(maskCap)
    index_offset[0] = get_timestamp_offset(imgCap)
    index_offset[1] = get_timestamp_offset(maskCap)

    centerOffsetAll = []
    st = time.time()

    for fIdx in indices:
        if fIdx == 0:
            get_frame_av(imgCap, fIdx - 0, index_offset[0], keyframe_interval[0])
            get_frame_av(maskCap, fIdx - 0, index_offset[1], keyframe_interval[1])
        else:
            get_frame_av(imgCap, fIdx - 1, index_offset[0], keyframe_interval[0])
            get_frame_av(maskCap, fIdx - 1, index_offset[1], keyframe_interval[1])

        # import pdb
        # pdb.set_trace()
        print("{}/{}".format(fIdx, max(indices)))
        for frame, mask  in zip(imgCap.decode(video=0), maskCap.decode(video=0)):

            img = np.asarray(frame.to_image().resize((IMAGE_W, IMAGE_H), Image.Resampling.LANCZOS)).copy()
            mask = np.asarray(mask.to_image().resize((IMAGE_W, IMAGE_H), Image.Resampling.LANCZOS)).copy()

            cropped_image, CENTER_OFFSET = cropImg(img[:,:,::-1], mask[:,:,::-1])
            centerOffsetAll.append([fIdx] + CENTER_OFFSET)
            makePath(osp.join(out_dir, 'imgs', str(fIdx).zfill(6)))
            cv2.imwrite(osp.join(out_dir, 'imgs', str(fIdx).zfill(6), f"image_c_{str(camIdx).zfill(3)}_f_{str(fIdx).zfill(6)}.png"), cropped_image)
            break

    centerOffsetAll = np.array(centerOffsetAll)
    makePath(osp.join(out_dir, 'centerOffsets'))
    pandas.DataFrame(centerOffsetAll).to_csv(osp.join(out_dir, 'centerOffsets', f"centerOffset_c_{camIdx}.csv"), header=None, index=None)
    print('Time: ', time.time() - st)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MetaCap")
    parser.add_argument("-si", '--slurm_id', default=0, type=int)
    parser.add_argument("-sn", "--slurm_num", default=1, type=int)
    parser.add_argument("-cn", "--cpu_num", default=9, type=int)

    parser.add_argument("-subi", "--subject_id", default=2, type=int)
    parser.add_argument("-ct", "--cloth_type", default='tight', type=str)
    parser.add_argument("-st", "--sequence_type", default='training', type=str)

    parser.add_argument("-cs", "--create_scene", default=False, action='store_true')
    parser.add_argument("-sc", "--select_camera", default=False, action='store_true')
    parser.add_argument("-in", "--index_name", default= 'training_indices_all.txt', type=str)

    args = parser.parse_args()

    slurm_id = args.slurm_id
    slurm_num = args.slurm_num
    create_scene = args.create_scene
    select_camera = args.select_camera
    numCpu = args.cpu_num

    subjectType = 'Subject{}'.format(str(args.subject_id).zfill(4))
    clothType = args.cloth_type
    sequenceType = args.sequence_type
    index_name  = sequenceType + '_indices.txt' #args.index_name

    base_dir = "/CT/HOIMOCAP4/work/data/WildDynaCap_official/{}/{}/{}".format(subjectType, clothType, sequenceType)
    cam_path = osp.join(base_dir, "cameras.calibration")
    idx_path = osp.join(base_dir, index_name)
    motion_dir = osp.join(base_dir, "motions")
    img_video_dir = osp.join(base_dir, "videos")
    mask_video_dir = osp.join(base_dir, 'masks')
    out_dir = makePath(osp.join(base_dir, 'recon_neus2'))
    Ks, Es, PsGL, Size = loadCamera(cam_path)
    print("\n + base_dir: {}".format(base_dir))
    print("+ cam_path: {}".format(cam_path))
    print("+ idx_path: {}".format(idx_path))
    print("+ img_video_dir: {}".format(img_video_dir))
    print("+ mask_video_dir: {}".format(mask_video_dir))
    print("+ out_dir: {}".format(out_dir))


    IMAGE_H, IMAGE_W = Size
    INTR = Ks
    EXTR = np.linalg.inv(Es)
    numCam =  len(Ks)
    if sequenceType in ['training', 'testing']:
        cropImgSize = [1600, 1200]
        imgSize = [3008, 4112]
    else:
        cropImgSize = [800, 600]
        imgSize = [1080, 1920]  # H,W

    # activeCamera = [47, 65, 75, 87]
    activeCamera = []

    camList = list(np.arange(numCam))
    camListInput = camList[slurm_id * numCam // slurm_num: (slurm_id + 1) * numCam // slurm_num]

    if os.path.exists(idx_path):
        indices = np.loadtxt(idx_path).astype('int').reshape(-1)
    else:
        print("Indices file does not exist: {}".format(idx_path))
        exit(0)

    print('Indices:\n')
    print(indices)

    print('select_camera: ', select_camera)
    print('create_scene: ', create_scene)
    if create_scene:
        motionPath = osp.join(base_dir, 'poseAngles.motion')
        if not os.path.isfile(motionPath):
            motionPath = osp.join(base_dir, 'motions', '54dof.motion')
        if not os.path.isfile(motionPath):
            motionPath = osp.join(base_dir, 'motions', '107dof.motion')
        print("+ motion_path: {}".format(motionPath))

        humanPosAll = loadMotion(motionPath)[indices,:3]
        centerOffsetAll = []
        for camIdx in range(0, numCam):
            temp = pandas.read_csv(osp.join(out_dir, 'centerOffsets', f"centerOffset_c_{camIdx}.csv"), index_col=None, header=None)
            centerOffsetAll.append(temp)
        centerOffsetAll = np.stack(centerOffsetAll, 0)  # C,F,H,W

        indices_csv = centerOffsetAll[0,:,:1].reshape(-1).tolist()
        for idx, fIdx in enumerate(indices_csv):
            print('{}/{}'.format(fIdx, max(indices_csv)), flush=True)
            output = init_params(humanPosAll[idx], H=cropImgSize[0], W= cropImgSize[1])

            cameras = []
            for camIdx in range(0, numCam):
                camera = {}
                camera['pose'] = EXTR[camIdx]
                camera['intrinsic'] = INTR[camIdx].copy()
                camera['intrinsic'][0][2] -= centerOffsetAll[camIdx][idx][2]
                camera['intrinsic'][1][2] -= centerOffsetAll[camIdx][idx][1]
                cameras.append(camera)

            if select_camera:
                output['frames'] = []
                for camIdx in activeCamera:
                    one_frame = {}
                    one_frame["file_path"] = 'image_c_' + str(camIdx) + '_f_' + str(fIdx) + '.png'
                    one_frame["transform_matrix"] = cameras[camIdx]['pose'].tolist()
                    ixt = cameras[camIdx]['intrinsic']
                    one_frame["intrinsic_matrix"] = ixt.tolist()
                    output['frames'].append(one_frame)

                out_json_path = osp.join(out_dir, 'imgs', str(fIdx).zfill(6), f'transform_{fIdx:06}_selectCamera.json')
                with open(out_json_path, 'w') as f:
                    json.dump(output, f, indent=4)
            else:
                output['frames'] = []
                for camIdx in range(numCam):
                    # add one_frame
                    one_frame = {}
                    one_frame["file_path"] = f"image_c_{str(camIdx).zfill(3)}_f_{str(fIdx).zfill(6)}.png"
                    one_frame["transform_matrix"] = cameras[camIdx]['pose'].tolist()
                    ixt = cameras[camIdx]['intrinsic']
                    one_frame["intrinsic_matrix"] = ixt.tolist()
                    output['frames'].append(one_frame)

                out_json_path = osp.join(out_dir, 'imgs', str(fIdx).zfill(6), f'transform_{fIdx:06}.json')
                with open(out_json_path, 'w') as f:
                    json.dump(output, f, indent=4)

    else:
        # import pdb
        # pdb.set_trace()
        for i in range(0, numCam // slurm_num // numCpu + 1):
            # if not i==12:
            #     continue
            camListInputFinal = camListInput[i * numCpu: (i + 1) * numCpu]
            print(camListInputFinal)
            p = Pool(numCpu)
            p.map(cropImgMask, camListInputFinal)