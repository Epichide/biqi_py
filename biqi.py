#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: EPICHIDE
# @Email: no email
# @Time: 2024/9/11 0:32
# @File: biqi.py
# @Software: PyCharm

import  cv2
import numpy as np
import pywt
from scipy.special import gamma

def biqi(image):
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

    for p, detail_coeffs in enumerate(coeffs):
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

    # 打印结果或其他处理
    print("Horizontal Gamma:", gam_horz)
    print("Vertical Gamma:", gam_vert)
    print("Diagonal Gamma:", gam_diag)
if __name__ == '__main__':
    image=cv2.imread("pepper_4.png")
    biqi(image)
