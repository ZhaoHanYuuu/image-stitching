"""
main中用到的一些函数

杜昱 2019K8009929015
duyu19@mails.ucas.ac.cn
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fontTools.misc.py23 import xrange


def rgb2hsv(rgb_img):
    b, g, r = cv2.split(rgb_img)
    height = rgb_img.shape[0]
    width = rgb_img.shape[1]
    H = np.zeros((height, width), np.float32)
    S = np.zeros((height, width), np.float32)
    V = np.zeros((height, width), np.float32)
    # r, g, b = cv2.split(rgb_img)
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    for i in range(0, height):
        for j in range(0, width):
            mx = max((b[i, j], g[i, j], r[i, j]))
            mn = min((b[i, j], g[i, j], r[i, j]))
            dt = mx - mn

            if mx == mn:
                H[i, j] = 0
            elif mx == r[i, j]:
                if g[i, j] >= b[i, j]:
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / dt)
                else:
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / dt) + 360
            elif mx == g[i, j]:
                H[i, j] = 60 * ((b[i, j]) - r[i, j]) / dt + 120
            elif mx == b[i, j]:
                H[i, j] = 60 * ((r[i, j]) - g[i, j]) / dt + 240
            H[i, j] = int(H[i, j] / 2)

            # S
            if mx == 0:
                S[i, j] = 0
            else:
                S[i, j] = int(dt / mx * 255)
            # V
            V[i, j] = int(mx * 255)
    return np.stack((H, S, V), axis=2)


def hsv2rgb(hsv_img):
    H, S, V = cv2.split(hsv_img)
    R2 = []
    G2 = []
    B2 = []
    for i in range(len(H)):
        R2.append([])
        G2.append([])
        B2.append([])
        for j in range(len(H[i])):
            if S[i][j] == 0:
                temp_R2 = V[i][j]
                temp_G2 = V[i][j]
                temp_B2 = V[i][j]
            else:
                temp_H = H[i][j] / 60
                case = round(temp_H)
                f = temp_H - i
                a = round(V[i][j] * (1 - S[i][j]))
                b = round(V[i][j] * (1 - S[i][j] * f))
                c = round(V[i][j] * (1 - S[i][j] * (1 - f)))
                if case == 0:
                    temp_R2 = V[i][j]
                    temp_G2 = c
                    temp_B2 = a
                elif case == 1:
                    temp_R2 = b
                    temp_G2 = V[i][j]
                    temp_B2 = a
                elif case == 2:
                    temp_R2 = a
                    temp_G2 = V[i][j]
                    temp_B2 = c
                elif case == 3:
                    temp_R2 = a
                    temp_G2 = b
                    temp_B2 = V[i][j]
                elif case == 4:
                    temp_R2 = c
                    temp_G2 = a
                    temp_B2 = V[i][j]
                elif case == 5:
                    temp_R2 = V[i][j]
                    temp_G2 = a
                    temp_B2 = b
            while temp_R2 not in range(256):
                if temp_R2 < 0:
                    temp_R2 = temp_R2 + 255
                else:
                    temp_R2 = temp_R2 - 255
            R2[i].append(temp_R2)
            while temp_G2 not in range(256):
                if temp_G2 < 0:
                    temp_G2 = temp_G2 + 255
                else:
                    temp_G2 = temp_G2 - 255
            G2[i].append(temp_G2)
            while temp_B2 not in range(256):
                if temp_B2 < 0:
                    temp_B2 = temp_B2 + 255
                else:
                    temp_B2 = temp_B2 - 255
            B2[i].append(temp_B2)
    return R2, G2, B2


def computeV(hsv_img):
    h, s, v = cv2.split(hsv_img)
    v = np.asarray(v, dtype="float32")
    return np.sum(v)
    # return np.mean(v)


def get_map(Hist):
    # 计算概率分布Pr
    sum_Hist = sum(Hist)
    Pr = Hist / sum_Hist
    # 计算累计概率Sk
    Sk = []
    temp_sum = 0
    for n in Pr:
        temp_sum = temp_sum + n
        Sk.append(temp_sum)
    Sk = np.array(Sk)
    # 计算映射关系img_map
    img_map = []
    for m in range(256):
        temp_map = int(255 * Sk[m] + 0.5)
        img_map.append(temp_map)
    img_map = np.array(img_map)
    return img_map


def get_off_map(map_):  # 计算反向映射，寻找最小期望
    map_2 = list(map_)
    off_map = []
    temp_pre = 0  # 如果循环开始就找不到映射时，默认映射为0
    for n in range(256):
        try:
            temp1 = map_2.index(n)
            temp_pre = temp1
        except BaseException:
            temp1 = temp_pre  # 找不到映射关系时，近似取向前最近的有效映射值
        off_map.append(temp1)
    off_map = np.array(off_map)
    return off_map


def get_infer_map(infer_img):
    infer_Hist_b = cv2.calcHist([infer_img], [0], None, [256], [0, 255])
    infer_Hist_g = cv2.calcHist([infer_img], [1], None, [256], [0, 255])
    infer_Hist_r = cv2.calcHist([infer_img], [2], None, [256], [0, 255])
    infer_b_map = get_map(infer_Hist_b)
    infer_g_map = get_map(infer_Hist_g)
    infer_r_map = get_map(infer_Hist_r)
    infer_b_off_map = get_off_map(infer_b_map)
    infer_g_off_map = get_off_map(infer_g_map)
    infer_r_off_map = get_off_map(infer_r_map)
    return [infer_b_off_map, infer_g_off_map, infer_r_off_map]


def get_finalmap(org_map, infer_off_map):  # 计算原始图像到最终输出图像的映射关系
    org_map = list(org_map)
    infer_off_map = list(infer_off_map)
    final_map = []
    for n in range(256):
        temp1 = org_map[n]
        temp2 = infer_off_map[temp1]
        final_map.append(temp2)
    final_map = np.array(final_map)
    return final_map


def get_newimg(img_org, org2infer_maps):
    w, h, _ = img_org.shape
    b, g, r = cv2.split(img_org)
    for i in range(w):
        for j in range(h):
            temp1 = b[i, j]
            b[i, j] = org2infer_maps[0][temp1]
    for i in range(w):
        for j in range(h):
            temp1 = g[i, j]
            g[i, j] = org2infer_maps[1][temp1]
    for i in range(w):
        for j in range(h):
            temp1 = r[i, j]
            r[i, j] = org2infer_maps[2][temp1]
    newimg = cv2.merge([b, g, r])
    return newimg


def get_new_img(img_org, infer_map):
    org_Hist_b = cv2.calcHist([img_org], [0], None, [256], [0, 255])
    org_Hist_g = cv2.calcHist([img_org], [1], None, [256], [0, 255])
    org_Hist_r = cv2.calcHist([img_org], [2], None, [256], [0, 255])
    org_b_map = get_map(org_Hist_b)
    org_g_map = get_map(org_Hist_g)
    org_r_map = get_map(org_Hist_r)
    org2infer_map_b = get_finalmap(org_b_map, infer_map[0])
    org2infer_map_g = get_finalmap(org_g_map, infer_map[1])
    org2infer_map_r = get_finalmap(org_r_map, infer_map[2])
    return get_newimg(img_org, [org2infer_map_b, org2infer_map_g, org2infer_map_r])


def style_transfer(image, ref):
    out = np.zeros_like(ref)
    _, _, ch = image.shape
    for i in range(ch):
        print(i)
        hist_img, _ = np.histogram(image[:, :, i], 256)
        hist_ref, _ = np.histogram(ref[:, :, i], 256)
        cdf_img = np.cumsum(hist_img)
        cdf_ref = np.cumsum(hist_ref)

        for j in range(256):
            tmp = abs(cdf_img[j] - cdf_ref)
            tmp = tmp.tolist()
            idx = tmp.index(min(tmp))  # 找出tmp中最小的数，得到这个数的索引
            out[:, :, i][ref[:, :, i] == j] = idx
    return out


def light(img, ref):
    out = np.zeros_like(img)
    _, _, colorChannel = img.shape
    for i in range(colorChannel):
        print(i)
        hist_img, _ = np.histogram(img[:, :, i], 256)  # get the histogram
        hist_ref, _ = np.histogram(ref[:, :, i], 256)
        cdf_img = np.cumsum(hist_img)  # get the accumulative histogram
        cdf_ref = np.cumsum(hist_ref)

        for j in range(256):
            tmp = abs(cdf_img[j] - cdf_ref)
            tmp = tmp.tolist()
            idx = tmp.index(min(tmp))  # find the smallest number in tmp, get the index of this number
            out[:, :, i][img[:, :, i] == j] = idx

    return out


def eqhist(histarray):
    farr = np.zeros(256,dtype= np.uint16)
    for i in range(256):
        farr[i] = np.sum(histarray[:i])*255
        if farr[i] > 255: farr[i] = 255
    return farr


def deal_light(img, ref):
    # imlist = []
    # imlist.append(img)
    # imlist.append(ref)
    srcIm = ref
    targIm = img
    srcIm.show()
    # targIm.show()

    srcbuf = np.array(srcIm)
    targbuf = np.array(targIm)

    # 显示原始直方图
    plt.subplot(2, 2, 1)
    srchist, bins, patche = plt.hist(srcbuf.flatten(), 256, normed = True)
    plt.subplot(2, 2, 3)
    targhist, bins, patche = plt.hist(targbuf.flatten(), 256, normed = True)

    # 先做直方图均衡
    resSrchist = eqhist(srchist)
    restarghist = eqhist(targhist)

    # 求匹配的映射关系序列
    MapArray = np.zeros(256,dtype= np.uint8)
    for x in xrange(256):
        MapArray[restarghist[x]] = x

    tmp = MapArray[0]
    for x in xrange(1,256):
        if MapArray[x] != 0:
            tmp = MapArray[x]
        else:
            MapArray[x] = tmp

    # 执行匹配
    outIm = srcIm.point(lambda i: MapArray[resSrchist[i]])
    plt.subplot(2, 2, 2)
    plt.hist(np.array(outIm).flatten(), 256, normed = True)

    return outIm