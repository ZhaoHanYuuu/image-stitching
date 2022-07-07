"""
sift算法具体实现

杜昱 2019K8009929015
duyu19@mails.ucas.ac.cn
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from functools import cmp_to_key


# 用于单通道图像
def print_img(name, img):
    plt.figure(name)
    image = cv2.merge([img, img, img])
    plt.imshow(image)
    plt.title(name)
    plt.show()


# 将彩色图像转换为灰度图像
def color_to_grey_img(color_img):
    if len(cv2.split(color_img)) == 3:
        b, g, r = cv2.split(color_img)
        grey_img = 0.299 * r + 0.587 * g + 0.114 * b
        return grey_img
    else:
        return color_img


# 用于将图片转化为float32
def img_uint8_to_float32(img):
    return img.astype(np.float32)


# 用于将图片转化为uint8
def img_float32_to_uint8(img):
    return img.astype(np.uint8)


def removeDuplicateSorted(keypoints):
    if len(keypoints) < 2:
        return keypoints
    # 进行排序
    keypoints.sort(key=cmp_to_key(keypoint_cmp))
    prev = 0
    clean_keypoints = [keypoints[prev]]
    for curr in range(1, len(keypoints)):
        if keypoint_unique(clean_keypoints[prev], keypoints[curr]):
            clean_keypoints.append(keypoints[curr])
            prev = prev + 1
    return clean_keypoints


# 该函数参考了opencv实现，对每个域都进行比较即可
# SIFT关键点是具有方向的圆形图像区域，由四个参数的几何框架描述
def keypoint_cmp(keypoint1, keypoint2):
    # pt, size, angle, response, octave
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] < keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] < keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size > keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle < keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response > keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave > keypoint1.octave
    return keypoint1.class_id > keypoint2.class_id


# 判断两个关键点的域是否相同
# SIFT关键点是具有方向的圆形图像区域，由四个参数的几何框架描述
def keypoint_unique(keypoint1, keypoint2):
    # pt, size, angle, response, octave
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return 1
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return 1
    if keypoint1.size != keypoint2.size:
        return 1
    if keypoint1.angle != keypoint2.angle:
        return 1
    return 0


# 计算组数与层数，尺度。从存储方式中解析出组数，层数，尺度
def unpackOctave(keypoint):
    # TODO
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave == 255:
        octave = -1
        scale = 2
    else:
        scale = 1 / (1 << octave)
    return octave, layer, scale


class MySift:
    # 使用David Lowe的参数
    def __init__(self, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
        # 中间层默认为3，增加这个数字可能会返回更精细的关键点，但可能因噪声使其选择变得不稳定。
        self.float_epsilon = 1e-7
        self.double_img_size = 1
        self.const_threshold = 0.04
        self.sigma = sigma
        self.num_intervals = num_intervals
        self.assumed_blur = assumed_blur
        self.image_border_width = image_border_width

    # 相当于sift的 main 函数
    def detectAndCompute(self, img, mask):
        print("    begin: detect and compute ...")
        if mask is not None:
            print("Error: Mask should be None.")
        # 将彩图转化为灰度图，修改数据类型，将图像扩大2倍
        init_img = self.createInitialImage(img)
        if self.double_img_size:
            # 高斯模糊，认为原图为0.5，需要到1.6
            # 由于图像扩大了一倍，因此其模糊程度为1
            sigma_diff = np.sqrt((self.sigma ** 2) - ((2 * self.assumed_blur) ** 2))
        else:
            sigma_diff = np.sqrt((self.sigma ** 2) - (self.assumed_blur ** 2))
        # 高斯模糊到 1.6
        image = cv2.GaussianBlur(init_img, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)
        print(f"    image shape is {image.shape}")
        # 计算高斯金字塔
        gaussian_imgs = self.buildGaussianPyramid(image)
        # 计算dog金字塔
        dog_imgs = self.buildDoGPyramid(gaussian_imgs)
        # 在dog尺度空间内找到极值点，存储相关信息
        kp_info = self.findScaleSpaceExtrema(dog_imgs)
        keypoints = []
        for kp in kp_info:
            # print("local extrema...")
            # 定位精准特征点
            local_result = self.adjustLocalExtrema(kp, dog_imgs)
            # 计算特征点的方向角度
            if local_result is not None:
                keypoint, local_index = local_result
                # 计算角度，并增加辅方向，返回一组关键点
                kps_or = self.calcOrientationHist(keypoint, local_index, kp[0], gaussian_imgs)
                # 将这组关键点加入
                for kp_or in kps_or:
                    keypoints.append(kp_or)
            # print("finish extrema")
        # 对重复项进行排序和删除
        keypoints = removeDuplicateSorted(keypoints)

        if self.double_img_size:
            # 将关键点从基础图像坐标转换为输入图像坐标，通过将相关属性减半实现
            for keypoint in keypoints:
                keypoint.pt = 0.5 * np.array(keypoint.pt)
                keypoint.size *= 0.5
                keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        # 生成描述符
        descriptors = self.calcSIFTDescriptors(keypoints, gaussian_imgs)
        # descriptors = np.array(descriptors)
        return keypoints, descriptors

    def createInitialImage(self, img):
        image = img_uint8_to_float32(img)
        grey_img = color_to_grey_img(image)
        # 图片扩大二倍
        # 最开始建立高斯金字塔时，要预先模糊输入图像来作为第0个组的第0层图像。相当于丢弃了最高的空域的采样率
        # 因此需要先将图像尺度扩大一倍生成-1组。
        if self.double_img_size:
            bigger_img = cv2.resize(grey_img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            return bigger_img
        else:
            return grey_img

    def buildGaussianPyramid(self, image):
        print("    begin: build gaussian pyramid ... ")
        octaves_num = int(np.log2(min(image.shape[0], image.shape[1])) - 2)  # 减2为了防止降采样过程中得到过小的函数
        gaussian_kernels = self.computeGaussianKernel()
        # 生成高斯图像，创建尺度空间金字塔
        gaussian_imgs = []
        next_octave_base = image
        for i_octave in range(octaves_num):
            octave_image = [next_octave_base]
            # 根据计算出的kernel进行计算
            for gaussian_sigma in gaussian_kernels[1:]:
                next_octave_base = cv2.GaussianBlur(next_octave_base, (0, 0), sigmaX=gaussian_sigma,
                                                    sigmaY=gaussian_sigma)
                octave_image.append(next_octave_base)
                # print_img("gaussian.jpg", img_float32_to_uint8(next_octave_base))
            gaussian_imgs.append(octave_image)
            # 降采样时，高斯金字塔上一组的初始图像来自于前一组的倒数第3张图像 σ(o, s) = σ_0 * 2^{s/S} , s = 0, 1, ... ,S+2
            next_octave_base = octave_image[-3]
            next_octave_base = cv2.pyrDown(next_octave_base)
        print("    end: finish generate gaussian images. ")
        return np.array(gaussian_imgs)

    # 不同组相同层的组内尺度坐标σ(s)相同。
    def computeGaussianKernel(self):
        gaussian_intervals_num = self.num_intervals + 3  # 是dog的S+3，dog是高斯两层相减得到的
        gaussian_kernels = np.zeros(gaussian_intervals_num)
        gaussian_kernels[0] = self.sigma
        # σ(o, s) = σ_0 * 2^{s/S} , s = 0, 1, ... ,S+2
        k = 2 ** (1 / (gaussian_intervals_num - 3))  # k = 2^{1/S}
        sigma_previous = self.sigma
        # 后一个 = 前一个 * k
        """
        在SITF的源码里，尺度空间里的每一层的图像（除了第1层）都是由其前面一层的图像和一个相对sigma的高斯滤波器卷积生成，
        而不是由原图和对应尺度的高斯滤波器生成的，这一方面是因为没有“原图”，
        输入图像I(x,y)已经是尺度为σ=0.5的图像了。另一方面是由于如果用原图计算，那么相邻两层之间相差的尺度实际上非常小，
        这样会造成在做高斯差分图像的时候，大部分值都趋近于0，以致于后面很难检测到特征点。
        """
        for s in range(1, gaussian_intervals_num):
            sigma_new = k * sigma_previous
            # 计算需要在前一个的基础上进行模糊的σ
            gaussian_kernels[s] = np.sqrt(sigma_new ** 2 - sigma_previous ** 2)
            sigma_previous = sigma_new
        return gaussian_kernels

    def buildDoGPyramid(self, gaussian_imgs):
        print("    begin: build DoG pyramid ... ")
        if self.num_intervals < 3:
            print(f"    Error: num_intervals too small, num_intervals = {self.num_intervals}")
        # todo: 使用并行计算
        dog_imgs = []
        for gaussian_octave_image in gaussian_imgs:
            dog_octave_image = []
            # 两个高斯图像做差
            for i_internal in range(len(gaussian_octave_image) - 1):
                # img_diff = cv2.absdiff()是错误的，必须使用cv2.subtract
                img_diff = cv2.subtract(gaussian_octave_image[i_internal], gaussian_octave_image[i_internal + 1])
                dog_octave_image.append(img_diff)
                # 查看打印dog图像
                # print_img("dog_img.jpg", img_float32_to_uint8(img_diff))
            dog_imgs.append(dog_octave_image)
        print("    end: finish generate dog images. ")
        return np.array(dog_imgs)

        # 初步探测极值点，邻域为26个点

    def findScaleSpaceExtrema(self, dog_imgs):
        print("    begin: find scale space extrema ...")
        print("           It takes several hours. ")
        kp_info = []
        # 使用两层循环，处理每个组以及每个组中的层之间DOG图像，获得关键点
        for i_octave in range(len(dog_imgs)):
            dog_octave_img = dog_imgs[i_octave]
            for i_interval in range(len(dog_octave_img) - 2):
                img1 = dog_octave_img[i_interval]
                img2 = dog_octave_img[i_interval + 1]
                img3 = dog_octave_img[i_interval + 2]
                # 边缘区域5个像素范围不被用来检测关键点
                print("      finding extrema ...")
                for i in range(self.image_border_width, img1.shape[0] - self.image_border_width):
                    for j in range(self.image_border_width, img1.shape[1] - self.image_border_width):
                        if self.isScaleSpaceExtrema(img1, img2, img3, i, j):
                            kp_info.append([i_octave, i_interval, i, j])
                            # print(f"octave_index {i_octave} image_index {i_interval} i {i} j {j}")
        print(f"        kp_info shape {np.array(kp_info).shape}")
        print("    end: finish get scale space extrema info.")
        return kp_info

    def isScaleSpaceExtrema(self, img1, img2, img3, i, j):
        # threshold = floor(0.5 ∗ contrast_threshold ∗ 255 ∗ SIFT_FIXPT_SCALE / nLayers)
        # 数据类型转换时已经缩放到1/255
        # 判断是否为极值点
        threshold = np.floor(0.5 * self.const_threshold / self.num_intervals * 255)
        current_point = img2[i, j]
        neighbor_points = []
        for m in range(3):
            for n in range(3):
                neighbor_points.append(img1[i - 1 + m, j - 1 + n])
                neighbor_points.append(img2[i - 1 + m, j - 1 + n])
                neighbor_points.append(img3[i - 1 + m, j - 1 + n])
        if abs(current_point) > threshold:
            if current_point > 0:
                return np.all(current_point >= neighbor_points)
            elif current_point < 0:
                return np.all(current_point <= neighbor_points)
        return False

    # 基于Lowe论文的第四部分，参考opencv实现 定位精准特征点
    def adjustLocalExtrema(self, kp, dog_imgs):
        img_scale = 1.0 / 255
        deriv_scale = img_scale * 0.5
        second_deriv_scale = img_scale
        cross_deriv_scale = img_scale * 0.25
        i_octave = kp[0]
        dog_images_in_octave = dog_imgs[kp[0]]
        i_img = kp[1] + 1
        i = kp[2]
        j = kp[3]
        image_shape = dog_images_in_octave[0].shape
        # 该参数来自lowe的论文
        r = 10
        num_attempts = 5
        # 初始化
        i_attempt = 0
        xi = 0.0
        xc = 0.0
        xr = 0.0
        gradient = [0.0, 0.0, 0.0]
        hessian = [[0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]]
        for i_attempt in range(num_attempts):
            img1 = dog_images_in_octave[i_img - 1]
            img2 = dog_images_in_octave[i_img]
            img3 = dog_images_in_octave[i_img + 1]
            gradient = [deriv_scale * (img2[i, j + 1] - img2[i, j + 1]),
                        deriv_scale * (img2[i + 1, j] - img2[i - 1, j]),
                        deriv_scale * (img3[i, j] - img1[i, j])]
            v2 = img2[i, j] * 2
            dxx = second_deriv_scale * (img2[i, j + 1] + img2[i, j - 1] - v2)
            dyy = second_deriv_scale * (img2[i + 1, j] + img2[i - 1, j] - v2)
            dss = second_deriv_scale * (img3[i, j] + img1[i, j] - v2)
            dxy = cross_deriv_scale * (img2[i + 1, j + 1] - img2[i + 1, j - 1] -
                                       img2[i - 1, j + 1] + img2[i - 1, j - 1])
            dxs = cross_deriv_scale * (img3[i, j + 1] - img3[i, j - 1] -
                                       img1[i, j + 1] + img1[i, j - 1])
            dys = cross_deriv_scale * (img3[i + 1, j] - img3[i - 1, j] -
                                       img1[i + 1, j] + img1[i - 1, j])
            hessian = [[dxx, dxy, dxs],
                       [dxy, dyy, dys],
                       [dxs, dys, dss]]
            # xc, xr, xi = -cv2.solve(np.array(hessian), np.array(gradient), cv2.DECOMP_LU)
            extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            xc = extremum_update[0]
            xr = extremum_update[1]
            xi = extremum_update[2]
            # print(f"xc {xc} xr {xr} xi {xi}")
            if abs(xc) < 0.5 and abs(xr) < 0.5 and abs(xi) < 0.5:
                break  # 已经收敛
            # 注意：此处必须加int
            j += int(round(xc))
            i += int(round(xr))
            i_img += int(round(xi))

            # 判断是否超出cube范围
            if i_img < 1 or i_img > self.num_intervals:
                return None
            if i < self.image_border_width or i >= image_shape[0] - self.image_border_width:
                return None
            if j < self.image_border_width or j >= image_shape[1] - self.image_border_width:
                return None
        # 次数够了还没收敛也直接返回空
        if i_attempt >= num_attempts:
            return None

        # 更新极值点处的函数值
        img2 = dog_images_in_octave[i_img + 1]
        contr = img2[i, j] * img_scale + np.dot(gradient, [xc, xr, xi])
        if abs(contr) * self.num_intervals < 0.04:
            return None
        tr = hessian[0][1] + hessian[1][1]  # dxx + dyy
        det = hessian[0][0] * hessian[1][1] - hessian[0][1] * hessian[0][1]  # dxx * dyy - dxy * dxy
        # 消除边缘响应，首先保证行列式的值大于0，然后判断变换相应，r=10
        # 此处要求det大于0是因为等于0时说明有为0的特征值，应舍去。
        # 小于0说明两个特征值一正一负，不符合对特征值描述函数变化快慢的性质的要求
        if det <= 0 or r * (tr * tr) >= ((r + 1) ** 2) * det:
            return None
        keypoint = cv2.KeyPoint()
        keypoint.pt = ((j + xc) * (1 << i_octave),
                       (i + xr) * (1 << i_octave))
        # 注意此处的存储方式
        keypoint.octave = i_octave + i_img * (1 << 8) + int(round((xi + 0.5) * 255)) * (1 << 16)
        keypoint.size = self.sigma * (2 ** ((i_img + xi) / np.float32(self.num_intervals))) * (1 << (i_octave + 1))
        # i_octave + 1 因为输入图像被扩大了，是原来的2倍
        keypoint.response = abs(contr)
        return keypoint, i_img

    # 计算梯度方向直方图
    def calcOrientationHist(self, keypoint, i_img, i_octave, gaussian_imgs):
        # opencv实现 SIFT_ORT_SIG_FCTR=1.5，方向分配的高斯sigma
        # 邻域尺度，3*1.5σ
        # 只统计3σ之内的，
        # 因为高斯分布的概率函数的99.7%位于三个标准差内
        # 故99.7%的权重位于关键点的3*scale像素尺度内
        # 在计算kpt.size时，使用了相对于第0层扩大一倍以后的初始图象，故此处需要乘0.5，再除以2^sigma
        scale = 1.5 * keypoint.size * 0.5 / (1 << i_octave)
        x_center = round(keypoint.pt[0] / (1 << i_octave))
        y_center = round(keypoint.pt[1] / (1 << i_octave))
        radius = round(3 * scale)
        # 使用36个方向的直方图
        raw_hist = np.zeros(36)
        smooth_hist = np.zeros(36)
        # 调整后的关键点所在的dog图像层，注意，此处不能勿用为调整前的
        gaussian_img = gaussian_imgs[i_octave][i_img]
        keypoints_new = []
        grad_collect = []
        # i的范围为 -radius ~ radius，
        for i in range(-radius, radius + 1):
            y = y_center + i
            if y < 0 or y >= gaussian_img.shape[0] - 1:
                continue
            for j in range(-radius, radius + 1):
                x = x_center + j
                if x <= 0 or x >= gaussian_img.shape[1] - 1:
                    continue

                dx = gaussian_img[y, x + 1] - gaussian_img[y, x - 1]
                dy = gaussian_img[y - 1, x] - gaussian_img[y + 1, x]
                # 高斯加权，离关键像素较远的 pixel 影响小
                weight = np.exp((i * i + j * j) * -1.0 / (2 * scale * scale))
                grad_collect.append([dx, dy, weight])
        # 计算直方图
        for dx, dy, weight in grad_collect:
            # 邻域中的所有像素
            grad_orientation = np.rad2deg(np.arctan2(dy, dx))
            histogram_index = int(round(grad_orientation * 36 / 360.))
            # 通过取余实现循环处理
            raw_hist[histogram_index % 36] += np.sqrt(dx * dx + dy * dy) * weight

        # 计算平滑后的直方图
        # 使用了36个柱，每个柱代表10度，因此需要对峰值位置进行插值，将最接近每一个
        # 峰值的直方图上的三个值拟合成抛物线
        for n in range(36):
            smooth_hist[n] = (raw_hist[n - 2] + raw_hist[(n + 2) % 36]) / 16. + (
                    raw_hist[n - 1] + raw_hist[(n + 1) % 36]) / 4. + 3 * raw_hist[n] / 8.
        # 为每个辅方向创建一个关键点
        # 论文中提到，这种额外的关键点在实际应用中显著有助于检测稳定性
        max_val = max(smooth_hist)
        mag_thr = 0.8 * max_val
        for k in range(36):
            l_index = k - 1
            r_index = (k + 1) % 36
            cur = smooth_hist[k]
            if cur > smooth_hist[l_index] and cur > smooth_hist[r_index] and cur >= mag_thr:
                left = smooth_hist[l_index]
                right = smooth_hist[r_index]
                index = (k + 0.5 * (left - right) / (left + right - 2 * cur)) % 36
                theta = 360 - index * 360 / 36
                # 除角度外，其他参数均与原关键点相同
                keypoint_new = cv2.KeyPoint()
                keypoint_new.pt = keypoint.pt
                keypoint_new.octave = keypoint.octave
                keypoint_new.size = keypoint.size
                keypoint_new.response = keypoint.response
                if abs(theta - 360) < self.float_epsilon:
                    keypoint_new.angle = 0
                else:
                    keypoint_new.angle = theta
                keypoints_new.append(keypoint_new)
        return keypoints_new

    # 计算生成描述子
    def calcSIFTDescriptors(self, keypoints, gaussian_images):
        # 描述符对关键点邻域的信息进行编码
        # lowe提出当梯度方向直方图是4*4维的时候，SIFT描述子具有很好得区分度
        # 每个梯度直方图有8个方向，一个方向代表45%
        # 4*4*8=128
        # 直方图宽度
        window_width = 4
        # 直方图有8个bin，用8个bin覆盖360度（与此前的36个bin区分）
        # 高斯加权函数的参数
        exp_scale = -1 / (0.5 * window_width * window_width)
        num_bins = 8
        bins_per_rad = num_bins / 360.
        # 将特征点附近邻域划分成 4 * 4 个子区域，每个子区域的尺寸为 3sigma
        # sigma为当前特征点的尺度值
        # 考虑到实际计算时，需要采用双线性插值，计算的图像区域为3sigma*(4+1)
        # 考虑旋转，应再乘根号2
        # scale_multiplier = 3
        descriptor_max_value = 0.2
        descriptors = []

        for keypoint in keypoints:
            # 提取当前特征点的所在层/组/尺度
            octave, layer, scale = unpackOctave(keypoint)
            gaussian_image = gaussian_images[octave + 1, layer]
            num_rows, num_cols = gaussian_image.shape
            point = np.round(scale * np.array(keypoint.pt)).astype('int')
            # attention，计算角度
            angle = 360. - keypoint.angle
            cos_angle = np.cos(np.deg2rad(angle))
            sin_angle = np.sin(np.deg2rad(angle))
            # 初始化
            row_list = []
            col_list = []
            magnitude_list = []
            orientation_list = []
            # 这里+2是为了处理插值
            histograms = np.zeros((window_width + 2, window_width + 2, num_bins))
            # 一个描述符方向直方图为3  3sigma
            # 公式见实验报告
            hist_width = 1.5 * scale * keypoint.size
            radius = round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5)
            # 避免半径过大，使其小于图像斜对角线长度
            radius = min(radius, np.sqrt(gaussian_image.shape[0] ** 2 + gaussian_image.shape[1] ** 2))

            for row in range(-radius, radius + 1):
                for col in range(-radius, radius + 1):
                    # 邻域中的点
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if 0 < window_row < gaussian_image.shape[0] - 1 and 0 < window_col < gaussian_image.shape[1] - 1:
                        # 保持旋转不变性，坐标轴旋转为关键点方向
                        row_rot = col * sin_angle + row * cos_angle
                        col_rot = col * cos_angle - row * sin_angle
                        # 把邻域区域的原点从中心位置移到该区域的左下角（+0.5d）,减0.5是进行坐标平移
                        row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                        col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                        if -1 < row_bin < window_width and -1 < col_bin < window_width:
                            # 差分求梯度，计算x和y的一阶导数，这里省略了分母2，是因为没有分母部分不影响后面的归一化处理
                            dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                            dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                            # 梯度幅值
                            gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                            # 梯度辐角
                            gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                            # 中间占比大，四周占比小，高斯加权函数
                            weight = np.exp(
                                exp_scale * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                            row_list.append(row_bin)
                            col_list.append(col_bin)
                            magnitude_list.append(weight * gradient_magnitude)
                            orientation_list.append((gradient_orientation - angle) * bins_per_rad)

            for row_bin, col_bin, magnitude, orientation_bin in zip(row_list, col_list, magnitude_list,
                                                                    orientation_list):
                # 参考OPENCV实现思路
                # 取三维坐标的整数部分，判断在4*4*8的区域中属于哪个正方体
                row_bin_floor = np.floor(row_bin).astype(int)
                col_bin_floor = np.floor(col_bin).astype(int)
                orientation_bin_floor = np.floor(orientation_bin).astype(int)
                # 取小数部分
                row_fraction = row_bin - row_bin_floor
                col_fraction = col_bin - col_bin_floor
                orientation_fraction = orientation_bin - orientation_bin_floor
                # 将0-360度以外的角度按照圆周循环调整回0-360
                if orientation_bin_floor < 0:
                    orientation_bin_floor += num_bins
                if orientation_bin_floor >= num_bins:
                    orientation_bin_floor -= num_bins
                # 按照三线性插值法，计算该像素对正方体的8个顶点的贡献大小
                r1 = magnitude * row_fraction
                r0 = magnitude - r1
                rc11 = r1 * col_fraction
                rc10 = r1 - rc11
                rc01 = r0 * col_fraction
                rc00 = r0 - rc01
                rco111 = rc11 * orientation_fraction
                rco110 = rc11 - rco111
                rco101 = rc10 * orientation_fraction
                rco100 = rc10 - rco101
                rco011 = rc01 * orientation_fraction
                rco010 = rc01 - rco011
                rco001 = rc00 * orientation_fraction
                rco000 = rc00 - rco001
                # 得到像素点在三维直方图中的索引
                ori_plus = (orientation_bin_floor + 1) % num_bins
                histograms[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += rco000
                histograms[row_bin_floor + 1, col_bin_floor + 1, ori_plus] += rco001
                histograms[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += rco010
                histograms[row_bin_floor + 1, col_bin_floor + 2, ori_plus] += rco011
                histograms[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += rco100
                histograms[row_bin_floor + 2, col_bin_floor + 1, ori_plus] += rco101
                histograms[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += rco110
                histograms[row_bin_floor + 2, col_bin_floor + 2, ori_plus] += rco111
            # 降到一维，展开
            descriptor_vector = histograms[1:-1, 1:-1, :].flatten()
            # 归一化，并将大于0.2的元素设置成0.2。
            # 为避免累加误差。0.2*平方和开方值，得到反归一化阈值thr
            thr = np.linalg.norm(descriptor_vector) * descriptor_max_value
            descriptor_vector[descriptor_vector > thr] = thr
            # 从float32转化到 unsigned char
            descriptor_vector /= max(np.linalg.norm(descriptor_vector), self.float_epsilon)
            descriptor_vector = np.round(512 * descriptor_vector)
            descriptor_vector[descriptor_vector < 0] = 0
            descriptor_vector[descriptor_vector > 255] = 255
            descriptors.append(descriptor_vector)
        return np.array(descriptors, dtype='float32')
