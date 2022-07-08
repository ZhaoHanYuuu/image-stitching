"""
UTF-8
中国科学院大学 数组图像处理2022 大作业-基础组-图像拼接
杜昱 2019K8009929015
duyu19@mails.ucas.ac.cn
"""
import numpy as np
import cv2
import mysift
import func

print("==============================")
print("DuYu 2019K8009929015")
print("DIP2022: image stitching")
img_seq = input("Enter the image sequence:")
while int(img_seq) < 1 or int(img_seq) > 4:
    print("Error: image sequence should be 1,2,3,4. ")
    img_seq = input("Enter the image sequence:")
img1_path = "./smaller_imgs/left" + img_seq + ".jpg"
img2_path = "./smaller_imgs/right" + img_seq + ".jpg"
print(f'Deal with {img1_path} and {img2_path} ...')
sift_test = 1
image_stitching = 2
model = image_stitching

if model is image_stitching:
    print("Reading images ...")
    # 读入图像
    left = cv2.imread(img1_path)
    right = cv2.imread(img2_path)

    print("Computing keypoints and descriptors ...")
    # 创建SIFT对象并获取关键点和描述子
    # sift = mysift.MySift()
    # 如果调用库函数：
    sift = cv2.SIFT_create()
    # 获取关键点和特征描述符
    left_kp, left_feature = sift.detectAndCompute(left, None)
    right_kp, right_feature = sift.detectAndCompute(right, None)

    print("Matching keypoints ...")
    # 关键点匹配
    # 创建一个暴力匹配器，返回DMatch对象列表。每个DMatch对象表示关键点的一个匹配结果
    # 尝试所有匹配从而找到最佳匹配
    # distance属性表示距离，距离越小匹配值越高
    # 查询描述符，训练描述符
    matcher = cv2.BFMatcher()
    feature_match = matcher.knnMatch(left_feature, right_feature, k=2)
    # des1为查询描述符，des2为训练描述符，k为返回的最佳匹配个数
    good_points = []
    good_matches = []
    # 应用比例测试选择要使用的匹配结果
    for m1, m2 in feature_match:
        if m1.distance < 0.75*m2.distance:
            good_points.append((m1.queryIdx, m1.trainIdx))
            good_matches.append([m1])
    match_img = cv2.drawMatchesKnn(left, left_kp, right, right_kp, good_matches, None, flags=2)
    middle_res_path = "./middle_res/before" + img_seq + ".jpg"
    cv2.imwrite(middle_res_path, match_img)
    print(f"    matching image is {middle_res_path}")

    # 根据筛选出的点重新确定关键点坐标
    left_good_kp = np.float32([left_kp[i].pt for (i, _) in good_points])
    right_good_kp = np.float32([right_kp[i].pt for (_, i) in good_points])
    # 计算单应矩阵，H为转换矩阵，status为mask掩码，标注出内点与外点
    H, status = cv2.findHomography(right_good_kp, left_good_kp, cv2.RANSAC, 5.0)

    print("Computing final image ...")

    # 计算图片尺寸
    left_height = left.shape[0]
    left_width = left.shape[1]
    right_width = right.shape[1]
    mix_height = left_height  # mix_height = max(l_h, r_h)
    mix_width = left_width + right_width
    # 计算左侧图片及其掩码
    left_mask_img = np.zeros((mix_height, mix_width, 3))
    # 过渡区域大小设置为20
    # offset = 600  # smoothing window size / 2
    # barrier = left_width - offset
    # mask = np.zeros((mix_height, mix_width))
    # # 过渡区域
    # mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset), (mix_height, 1))
    # mask[:, :barrier - offset] = 1
    # # 将掩码变成 3-通道的
    # left_mask = np.stack((mask, mask, mask), axis=2)
    # left_mask_img[0:left_height, 0:left_width, :] = left
    # left_mask_img = left_mask_img * left_mask
    # 右侧图片及其掩码
    # right_hsv = func.rgb2hsv(right)
    # right_hsv = cv2.cvtColor(right, cv2.COLOR_BGR2HSV)
    # right_v = func.computeV(right_hsv)
    # # left_hsv = func.rgb2hsv(left)
    # left_hsv = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)
    # left_v = func.computeV(left_hsv)
    # bright_k = left_v / right_v
    # # bright_k = (bright_k - 1) / 2 + 1
    # print(bright_k)
    # h, s, v = cv2.split(right_hsv)
    # v = np.asarray(v, dtype="float32")
    # v = v / bright_k
    # v_new = np.round(v)
    # v_new = np.where(v_new > 255, 255, v_new)
    # v_new = np.where(v_new < 0, 0, v_new)
    # v_new = np.asarray(v_new, dtype="uint8")
    # right_hsv = cv2.merge([h, s, v_new])
    # right = cv2.cvtColor(right_hsv, cv2.COLOR_HSV2BGR)

    # infer_map = func.get_infer_map(left)
    # right = func.get_new_img(right, infer_map)

    # right = func.style_transfer(right, left)

    mask2 = np.zeros((mix_height, mix_width))
    # mask2[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset), (mix_height, 1))
    # mask2[:, barrier + offset:] = 1
    # right_mask = np.stack((mask2, mask2, mask2), axis=2)

    # 透视变换
    # right_mask_img = cv2.warpPerspective(right, H, (mix_width, mix_height)) * right_mask
    right_mask_img = cv2.warpPerspective(right, H, (mix_width, mix_height))
    # cv2.imshow("11", right_mask_img)
    # cv2.imwrite("check.jpg", right_mask_img)  # 写入文件
    zero_test = np.nonzero(right_mask_img)
    min_index = zero_test[0][0]
    row, col = np.where(right_mask_img[:, :, 0] != 0)
    offset = (left_width - max(col)) * 0.2
    offset = int(offset)
    print(offset)
    barrier = left_width - offset
    mask = np.zeros((mix_height, mix_width))
    # 过渡区域
    mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset), (mix_height, 1))
    mask[:, :barrier - offset] = 1
    # 将掩码变成 3-通道的
    left_mask = np.stack((mask, mask, mask), axis=2)
    left_mask_img[0:left_height, 0:left_width, :] = left
    left_mask_img = left_mask_img * left_mask

    mask2[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset), (mix_height, 1))
    mask2[:, barrier + offset:] = 1
    right_mask = np.stack((mask2, mask2, mask2), axis=2)

    right_mask_img = right_mask_img * right_mask
    cv2.imwrite("check.jpg", right_mask_img)  # 写入文件
    # 合并左图和右图
    cv2.imwrite("left.jpg", left_mask_img)  # 写入文件
    mix_img = left_mask_img + right_mask_img
    # 裁剪掉右侧的黑边
    rows, cols = np.where(mix_img[:, :, 0] != 0)
    output_img = mix_img[min(rows):max(rows), min(cols):max(cols), :]
    # 显示输出图像

    result_path = "./result/output" + img_seq + ".jpg"
    cv2.imwrite(result_path, output_img)  # 写入文件
    print(f"The final result is {result_path}")

# 测试SIFT特征点检测是否正常
elif model is sift_test:
    # 读入图像
    left = cv2.imread("./smaller_imgs/small.jpg")
    sift = mysift.MySift()
    left_kp, left_feature = sift.detectAndCompute(left, None)  # numpy.ndarray，二维列表， x*128
    cv2.drawKeypoints(left, left_kp, left)
    cv2.imshow('final output', left)  # 显示图片
    cv2.waitKey(0)

print("==============================")

