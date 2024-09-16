#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: EPICHIDE
# @Email: no email
# @Time: 2024/9/11 0:32
# @File: biqi.py
# @Software: PyCharm
import numpy as np
from scipy.ndimage import convolve
import  cv2
import numpy as np
import pywt
from scipy.special import gamma
import numpy as np
import os
import subprocess

def biqi(image):
    rep_vec_27=biqi_feature(image)

    # 移除均值0-8
    rep_vec = rep_vec_27[9:]
    rep_vec=rep_vec.reshape(1,-1)
    # 分类
    with open('test_ind.txt', 'w') as fid:
        for j in range(rep_vec.shape[0]):
            fid.write(str(j+1) + ' ')
            for k in range(rep_vec.shape[1]):
                fid.write(str(k+1) + ':' + str(rep_vec[j, k]) + ' ')
            fid.write('\n')

    # 调用系统命令进行svm-scale和svm-predict
    subprocess.run(['svm-scale', '-r', 'range2', 'test_ind.txt', '>>', 'test_ind_scaled'])
    subprocess.run(['svm-predict', '-b', '1', 'test_ind_scaled', 'model_89', 'output_89'])
    os.remove('test_ind.txt')
    os.remove('test_ind_scaled')

    # 质量评估
    with open('test_ind.txt', 'w') as fid:
        for j in range(rep_vec.shape[0]):
            fid.write(str(j+1) + ' ')
            for k in range(rep_vec.shape[1]):
                fid.write(str(k+1) + ':' + str(rep_vec[j, k]) + ' ')
            fid.write('\n')

    # Jp2k 质量

    subprocess.run(['svm-scale', '-r', 'range2_jp2k', 'test_ind.txt', '>>', 'test_ind_scaled'])
    subprocess.run(['svm-predict', '-b', '1', 'test_ind_scaled', 'model_89_jp2k', 'output_blur'])
    jp2k_score = np.genfromtxt('output_blur', delimiter=' ')  # 假设输出被保存为格式
    os.remove('output_blur')
    os.remove('test_ind_scaled')

    # JPEG 质量
    jpeg_score = jpeg_quality_score(image)[0]  # 假设jpeg_quality_score是一个已定义的函数

    # WN 质量
    subprocess.run(['svm-scale', '-r', 'range2_wn', 'test_ind.txt', '>>', 'test_ind_scaled'])
    subprocess.run(['svm-predict', '-b', '1', 'test_ind_scaled', 'model_89_wn', 'output_blur'])
    wn_score = np.loadtxt('output_blur')
    os.remove('output_blur')
    os.remove('test_ind_scaled')

    # 模糊质量
    subprocess.run(['svm-scale', '-r', 'range2_blur', 'test_ind.txt', '>>', 'test_ind_scaled'])
    subprocess.run(['svm-predict', '-b', '1', 'test_ind_scaled', 'model_89_blur', 'output_blur'])
    blur_score = np.loadtxt('output_blur')
    os.remove('output_blur')
    os.remove('test_ind_scaled')

    # FF 质量
    subprocess.run(['svm-scale', '-r', 'range2_ff', 'test_ind.txt', '>>', 'test_ind_scaled'])
    subprocess.run(['svm-predict', '-b', '1', 'test_ind_scaled', 'model_89_ff', 'output_blur'])
    ff_score = np.loadtxt('output_blur')
    os.remove('output_blur')
    os.remove('test_ind_scaled')
    os.remove('test_ind.txt')

    # 最终聚合
    with open('output_89', 'r') as fid:
        next(fid)  # 跳过标题行
        C = np.loadtxt(fid)
        probs = C[1:]
    scores = np.array([jp2k_score, jpeg_score, wn_score, blur_score, ff_score])
    quality = np.sum(probs * scores)

    # 清理文件
    os.remove('output_89')
    return quality

def biqi_feature(image):
    if image.ndim==3:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    num_scales=3
    gam=np.arange(0.2,10,0.001)
    r_gam=gamma(1/gam)*gamma(3/gam)/(gamma(2/gam)**2)
    S=pywt.wavedec2(image, wavelet="db9", level = num_scales)
    print()


    # 假设image是已经加载的图像
    # image = ...

    num_scales = 3
    gam = np.arange(0.2, 10.001, 0.001)
    r_gam = gamma(1. / gam) * gamma(3. / gam) / (gamma(2. / gam) ** 2)

    coeffs = pywt.wavedec2(image, 'db9', level=num_scales)
    cA = coeffs[0]  # 近似系数
    coeffs = coeffs[1:]  # 细节系数列表

    horz = []
    vert = []
    diag = []
    mu_horz = []
    sigma_sq_horz = []
    E_horz = []
    rho_horz = []
    gam_horz = []
    mu_vert = []
    sigma_sq_vert = []
    E_vert = []
    rho_vert = []
    gam_vert = []
    mu_diag = []
    sigma_sq_diag = []
    E_diag = []
    rho_diag = []
    gam_diag = []

    for p, detail_coeffs in enumerate(coeffs[::-1]):
        horz_temp, vert_temp, diag_temp = detail_coeffs
        horz.append(horz_temp.ravel())
        diag.append(diag_temp.ravel())
        vert.append(vert_temp.ravel())

        # 计算统计量
        h_horz_curr = np.array(horz[-1])
        h_vert_curr = np.array(vert[-1])
        h_diag_curr = np.array(diag[-1])

        mu_horz.append(np.mean(h_horz_curr))
        sigma_sq_horz.append(np.mean((h_horz_curr - mu_horz[-1]) ** 2))
        E_horz.append(np.mean(np.abs(h_horz_curr - mu_horz[-1])))
        rho_horz.append(sigma_sq_horz[-1] / E_horz[-1] ** 2)
        array_position = np.argmin(np.abs(rho_horz[-1] - r_gam))
        min_difference = np.min(np.abs(rho_horz[-1] - r_gam))

        gam_horz.append(gam[array_position])

        mu_vert.append(np.mean(h_vert_curr))
        sigma_sq_vert.append(np.mean((h_vert_curr - mu_vert[-1]) ** 2))
        E_vert.append(np.mean(np.abs(h_vert_curr - mu_vert[-1])))
        rho_vert.append(sigma_sq_vert[-1] / E_vert[-1] ** 2)
        array_position = np.argmin(np.abs(rho_vert[-1] - r_gam))
        min_difference = np.min(np.abs(rho_vert[-1] - r_gam))

        gam_vert.append(gam[array_position])
        mu_diag.append(np.mean(h_diag_curr))
        sigma_sq_diag.append(np.mean((h_diag_curr - mu_diag[-1]) ** 2))
        E_diag.append(np.mean(np.abs(h_diag_curr - mu_diag[-1])))
        rho_diag.append(sigma_sq_diag[-1] / E_diag[-1] ** 2)
        array_position = np.argmin(np.abs(rho_diag[-1] - r_gam))
        min_difference = np.min(np.abs(rho_diag[-1] - r_gam))
        gam_diag.append(gam[array_position])
    rep_vec=np.concatenate([mu_horz,mu_vert,mu_diag,
                            sigma_sq_horz,sigma_sq_vert,sigma_sq_diag,
                            gam_horz,gam_vert,gam_diag],axis=0)

    # 打印结果或其他处理
    # print("Horizontal Gamma:", gam_horz)
    # print("Vertical Gamma:", gam_vert)
    # print("Diagonal Gamma:", gam_diag)
    return  rep_vec





def jpeg_quality_score(img):
    """
    # Example usage:
    # img = np.array(...)  # Load your image here
    # score, B, A, Z = jpeg_quality_score(img)
    This function calculates the quality score of a JPEG compressed image.
    Input: A test 8-bit per pixel grayscale image loaded in a 2-D array
    Output: A quality score of the image. The score typically has a value
            between 1 and 10 (10 represents the best quality, 1 the worst).
    """
    if img.ndim==3:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if img.ndim != 2:
        score = -1
        return score, None, None, None

    M, N = img.shape
    if M < 16 or N < 16:
        score = -2
        return score, None, None, None

    x = img.astype(np.float64)

    # Feature Extraction:
    # 1. Horizontal features
    d_h = np.diff(x, axis=1)
    B_h = np.mean(np.abs(d_h[:, 8::8][:, :(N // 8) - (N % 8 == 0)]))
    A_h = (8 * np.mean(np.abs(d_h)) - B_h) / 7
    sig_h = np.sign(d_h)
    Z_h = np.mean((sig_h[:, :-1] * sig_h[:, 1:]) < 0)

    # 2. Vertical features
    d_v = np.diff(x, axis=0)
    B_v = np.mean(np.abs(d_v[8::8, :][:, :(M // 8) - (M % 8 == 0)]))
    A_v = (8 * np.mean(np.abs(d_v)) - B_v) / 7
    sig_v = np.sign(d_v)
    Z_v = np.mean((sig_v[:-1, :] * sig_v[1:, :]) < 0)

    # 3. Combined features
    B = (B_h + B_v) / 2
    A = (A_h + A_v) / 2
    Z = (Z_h + Z_v) / 2

    # Quality Prediction
    alpha = -927.4240
    beta = 850.8986
    gamma1 = 235.4451
    gamma2 = 128.7548
    gamma3 = -341.4790
    score = alpha + beta * (B ** (gamma1 / 10000)) * (A ** (gamma2 / 10000)) * (Z ** (gamma3 / 10000))

    return score, B, A, Z



if __name__ == '__main__':
    image=cv2.imread("pepper_4.png")
    quality=biqi(image)
    print(quality)