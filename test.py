import cv2
import numpy as np
import os
import torch
import time
from PIL import Image, ImageOps
from Fusionnet import MACTFusion as net

# from loss import Fusionloss

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = net(output=1)
model_path = "./modelpath/cross/fusion_model.pth"
use_gpu = torch.cuda.is_available()

if use_gpu:
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
else:
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)


def imresize(arr, size, interp='bilinear', mode=None):
    numpydata = np.asarray(arr)
    im = Image.fromarray(numpydata, mode=mode)
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size) * percent).astype(int))
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size) * size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])
    return np.array(imnew)


def resize(image1, image2, crop_size_img, crop_size_label):
    image1 = imresize(image1, crop_size_img, interp='bicubic')
    image2 = imresize(image2, crop_size_label, interp='bicubic')
    return image1, image2


def get_image_files(input_folder):
    # 获取指定文件夹中的所有图像文件
    valid_extensions = (".bmp", ".tif", ".jpg", ".jpeg", ".png")
    return sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)])


def fusion(input_folder_ir, input_folder_vis, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tic = time.time()
    # criteria_fusion = Fusionloss()

    ir_images = get_image_files(input_folder_ir)
    vis_images = get_image_files(input_folder_vis)

    for ir_image, vis_image in zip(ir_images, vis_images):
        path1 = os.path.join(input_folder_ir, ir_image)
        path2 = os.path.join(input_folder_vis, vis_image)

        # 读取灰度图像
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        # 调整图像大小
        img1, img2 = resize(img1, img2, [256, 256], [256, 256])

        # 归一化图像
        img1 = np.asarray(img1, dtype=np.float32) / 255.0
        img2 = np.asarray(img2, dtype=np.float32) / 255.0

        # 扩展维度
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)

        # 转换为张量
        img1_tensor = torch.from_numpy(img1).unsqueeze(0).to(device)
        img2_tensor = torch.from_numpy(img2).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            out = model(img1_tensor, img2_tensor)

            # 将张量转换为 NumPy 数组
            out_np = out.cpu().numpy()

            # 归一化输出
            out_np = (out_np - np.min(out_np)) / (np.max(out_np) - np.min(out_np))

        # 处理和保存结果
        d = np.squeeze(out_np)
        result = (d * 255).astype(np.uint8)

        # 获取文件扩展名，并保持与输入文件相同的格式
        output_filename = os.path.splitext(ir_image)[0] + os.path.splitext(ir_image)[1]
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, result)

    toc = time.time()
    print('Processing time: {}'.format(toc - tic))


if __name__ == '__main__':
    input_folder_1 = 'path1'
    input_folder_2 = 'path2'
    output_folder = './fusion'

    fusion(input_folder_2, input_folder_1, output_folder)
