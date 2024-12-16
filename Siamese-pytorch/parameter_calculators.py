# 各个指标的特征
## RGB
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_color_histogram(image_path, bins=8):
    # 加载图像并转换为RGB格式
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 计算每个通道的颜色直方图
    hist_r = cv2.calcHist([image_rgb], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([image_rgb], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([image_rgb], [2], None, [bins], [0, 256])

    # 将三个直方图拼接成一个特征向量
    color_hist_feature = np.concatenate((hist_r.flatten(), hist_g.flatten(), hist_b.flatten()))

    plt.figure()
    plt.title("Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    
    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        histr = cv2.calcHist([image_rgb], [i], None, [bins], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, bins])
    
    plt.show()

    return color_hist_feature

# 使用函数
histogram_feature = calculate_color_histogram('/home/hechunjiang/gradio/样品1 LG 65UF8580/华为P50手机采集图像/监视器采集图像/1.jpg', bins=16)
print("Histogram feature vector:", histogram_feature)
## HSV
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_hsv_histogram(image_path, bins=8):
    # 加载图像并转换为HSV格式
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 计算每个通道的颜色直方图
    hist_h = cv2.calcHist([image_hsv], [0], None, [bins], [0, 180])  # H通道范围是[0, 180]
    hist_s = cv2.calcHist([image_hsv], [1], None, [bins], [0, 256])  # S通道范围是[0, 256]
    hist_v = cv2.calcHist([image_hsv], [2], None, [bins], [0, 256])  # V通道范围是[0, 256]

    # 将三个直方图拼接成一个特征向量
    hsv_hist_feature = np.concatenate((hist_h.flatten(), hist_s.flatten(), hist_v.flatten()))

    # 可视化直方图（可选）
    plt.figure()
    plt.title("HSV Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    colors = ('r', 'g', 'b')
    channel_names = ('Hue', 'Saturation', 'Value')
    for i, col in enumerate(colors):
        histr = cv2.calcHist([image_hsv], [i], None, [bins], [0, 180 if i == 0 else 256])
        plt.plot(histr, color=col, label=channel_names[i])
        plt.xlim([0, bins])
    
    plt.legend()
    plt.show()

    return hsv_hist_feature

# 使用函数
histogram_feature = calculate_hsv_histogram('/home/hechunjiang/gradio/样品1 LG 65UF8580/华为P50手机采集图像/监视器采集图像/1.jpg', bins=16)
print("HSV Histogram feature vector:", histogram_feature)
## 灰度值
def calculate_grayscale_histogram(image_path, bins=256):
    # 加载图像并转换为灰度格式
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return
    
    # 计算灰度直方图
    hist_gray = cv2.calcHist([image], [0], None, [bins], [0, 256])

    # # 将直方图归一化（可选）
    # hist_gray_normalized = cv2.normalize(hist_gray, hist_gray).flatten()

    # 可视化直方图（可选）
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist_gray, color='gray')
    plt.xlim([0, bins])
    plt.show()

    return hist_gray

# 使用函数
histogram_feature = calculate_grayscale_histogram('/home/hechunjiang/gradio/样品1 LG 65UF8580/华为P50手机采集图像/监视器采集图像/1.jpg', bins=16)
print("Grayscale histogram feature vector:", histogram_feature)
## 颜色矩
import cv2
import numpy as np

def calculate_color_moments(image_path):
    # 加载图像并转换为RGB格式
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 初始化一个列表用于存储所有颜色通道的颜色矩
    color_moments = []

    # 分别对R, G, B三个通道进行计算
    for channel_id in range(3):
        channel = image_rgb[:, :, channel_id]

        # 计算均值（Mean）
        mean = np.mean(channel)

        # 计算标准差（Standard Deviation）
        std_dev = np.std(channel)

        # 计算偏度（Skewness），需要先减去均值再除以标准差
        skewness = np.mean(((channel - mean) / std_dev) ** 3) if std_dev != 0 else 0

        # 将当前通道的颜色矩添加到列表中
        color_moments.extend([mean, std_dev, skewness])

    # 返回颜色矩作为一个特征向量
    return np.array(color_moments)

# 使用函数
color_moments_features = calculate_color_moments('/home/hechunjiang/gradio/样品1 LG 65UF8580/华为P50手机采集图像/监视器采集图像/1.jpg')
print("Color Moments (均值, 方差, 偏度 for R, G, B):", color_moments_features)
## GLCM
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt


def calculate_glcm_features(image_path, distances=[1], angles=[0], levels=256, props=['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']):
    # 加载图像并转换为灰度格式
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return

    # 计算灰度共生矩阵
    glcm = graycomatrix(image, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)

    # 提取指定属性的特征值
    features = []
    for prop in props:
        feature = graycoprops(glcm, prop)
        features.append(feature.flatten())

    # 将所有特征合并成一个特征向量
    glcm_features = np.concatenate(features)

    # 可视化灰度图像（可选）
    plt.figure()
    plt.title("Grayscale Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

    # 返回GLCM特征向量
    return glcm_features


# 使用函数
glcm_features = calculate_glcm_features(
    '/home/hechunjiang/gradio/样品1 LG 65UF8580/华为P50手机采集图像/监视器采集图像/1.jpg', distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
print("GLCM Features:", glcm_features)
## LBP
import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt

def calculate_lbp_features(image_path, P=8, R=1, method='uniform'):
    # 加载图像并转换为灰度格式
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return
    
    # 计算LBP特征
    lbp = feature.local_binary_pattern(image, P, R, method)

    # 计算LBP直方图
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))

    # 归一化直方图
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # 避免除以零

    # 可视化LBP图像（可选）
    plt.figure()
    plt.title("LBP Image")
    plt.imshow(lbp, cmap='gray')
    plt.axis('off')
    plt.show()

    # 可视化LBP直方图（可选）
    plt.figure()
    plt.title("LBP Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, len(hist)])
    plt.show()

    # 返回LBP直方图作为一个特征向量
    return hist

# 使用函数
lbp_features = calculate_lbp_features('/home/hechunjiang/gradio/样品1 LG 65UF8580/华为P50手机采集图像/监视器采集图像/1.jpg', P=8, R=1, method='uniform')
print("LBP Features:", lbp_features)
## 小波变换
import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

def calculate_wavelet_features(image_path, wavelet='db1', level=1):
    # 加载图像并转换为灰度格式
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return
    
    # 对图像进行小波变换
    coeffs = pywt.wavedec2(image, wavelet, level=level)

    # 提取近似系数和其他细节系数
    approx_coeffs = coeffs[0]
    detail_coeffs = coeffs[1:]

    # 将所有系数展平成一维数组
    all_coeffs = [approx_coeffs.flatten()]
    for d in detail_coeffs:
        for band in d:
            all_coeffs.append(band.flatten())

    # 将所有系数合并成一个特征向量
    wavelet_features = np.concatenate(all_coeffs)

    # 可视化小波变换结果（可选）
    plt.figure(figsize=(12, 8))
    
    # 显示原始图像
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    # 显示近似系数
    plt.subplot(2, 3, 2)
    plt.title("Approximation Coefficients")
    plt.imshow(approx_coeffs, cmap='gray')
    plt.axis('off')

    # 显示水平、垂直和对角细节系数
    detail_bands = ['Horizontal', 'Vertical', 'Diagonal']
    for i, (d, label) in enumerate(zip(detail_coeffs[-1], detail_bands)):
        plt.subplot(2, 3, 3 + i)
        plt.title(f"{label} Detail Coefficients")
        plt.imshow(d, cmap='gray')
        plt.axis('off')

    plt.show()

    # 返回小波变换的特征向量
    return wavelet_features

# 使用函数
wavelet_features = calculate_wavelet_features('/home/hechunjiang/gradio/样品1 LG 65UF8580/华为P50手机采集图像/监视器采集图像/1.jpg', wavelet='haar', level=1)
print("Wavelet Features:", wavelet_features)
## Hu
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_hu_moments(image_path):
    # 加载图像并转换为灰度格式
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return
    
    # 应用二值化处理，确保只计算前景对象的Hu矩
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 计算图像矩
    moments = cv2.moments(binary_image)

    # 计算Hu矩
    hu_moments = cv2.HuMoments(moments).flatten()

    # 可视化二值化后的图像（可选）
    plt.figure()
    plt.title("Binary Image")
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')
    plt.show()

    # 返回Hu矩作为一个特征向量
    return hu_moments

# 使用函数
hu_moments_features = calculate_hu_moments('/home/hechunjiang/gradio/样品1 LG 65UF8580/华为P50手机采集图像/监视器采集图像/1.jpg')
print("Hu Moments Features:", hu_moments_features)
## HOG特征
import cv2
import numpy as np


def calculate_hog_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像

    if image is None:
        print("Error: Image not found.")
        exit()

    # 调整图像大小（HOG需要统一的图像大小）
    image = cv2.resize(image, (128, 64))  # 常用的HOG输入大小

    # 初始化 HOG 描述符
    win_size = (128, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    # 计算 HOG 特征
    hog_features = hog.compute(image)

    hog_features_array = hog_features.flatten()

    return hog_features_array

# 加载图像并进行预处理
image_path = '/home/hechunjiang/gradio/样品1 LG 65UF8580/华为P50手机采集图像/监视器采集图像/1.jpg'  # 替换为你的图像路径

hog_features_array = calculate_hog_features(image_path)
print(f"hog features shape: {hog_features_array.shape}, features: {hog_features_array}")
## zernike矩
import cv2
import numpy as np
import mahotas
import mahotas.features
import matplotlib.pyplot as plt

def calculate_zernike_moments(image_path, radius=64, degree=8):
    # 加载图像并转换为灰度格式
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return
    
    # 应用二值化处理，确保只计算前景对象的Zernike矩
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 查找轮廓并找到最大轮廓（假设是主要对象）
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    # 获取最小外接圆以确定中心点和半径
    (center_x, center_y), min_enclosing_radius = cv2.minEnclosingCircle(max_contour)
    center = (int(center_x), int(center_y))
    min_enclosing_radius = int(min_enclosing_radius)

    # 创建掩码并应用到二值化图像上，确保只有感兴趣区域被考虑
    mask = np.zeros_like(binary_image)
    cv2.circle(mask, center, min_enclosing_radius, 255, -1)
    masked_image = cv2.bitwise_and(binary_image, binary_image, mask=mask)

    # 计算Zernike矩
    zernike_moments = mahotas.features.zernike(masked_image, radius=min_enclosing_radius, degree=degree)

    # 可视化二值化后的图像及其掩码（可选）
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Binary Image")
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Masked Image with Circle")
    plt.imshow(masked_image, cmap='gray')
    plt.scatter(center_x, center_y, c='red', marker='x')  # 标记中心点
    plt.axis('off')

    plt.show()

    # 返回Zernike矩作为一个特征向量
    return zernike_moments, masked_image

# 使用函数
zernike_features, masked_image = calculate_zernike_moments('/home/hechunjiang/gradio/样品1 LG 65UF8580/华为P50手机采集图像/监视器采集图像/1.jpg', radius=64, degree=8)
print("Zernike Moments Features:", zernike_features)
## 边缘
import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_edge_detection(image_path, low_threshold=100, high_threshold=200):
    # 加载图像并转换为灰度格式
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return
    
    # 高斯模糊以减少噪声
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Canny 边缘检测
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

    # 可视化原始图像和边缘检测结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Canny Edge Detection")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.show()

    # 返回边缘图像
    return edges

# 使用函数
edges = canny_edge_detection('/home/hechunjiang/gradio/样品1 LG 65UF8580/华为P50手机采集图像/监视器采集图像/1.jpg')
print("Edges detected.")

# 如果需要将边缘检测结果作为特征向量，可以将边缘图像展平成一维数组
edge_features = edges.flatten()
print("Edge Features:", edge_features)

## SIFT
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_sift_features(image_path):
    # 加载图像并转换为灰度格式
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return
    
    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 检测关键点并计算描述符
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # 可视化关键点（可选）
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Image with SIFT Keypoints")
    plt.imshow(image_with_keypoints)
    plt.axis('off')

    plt.show()

    # 返回SIFT特征向量（描述符）
    return keypoints, descriptors

# 使用函数
keypoints, descriptors = calculate_sift_features('/home/hechunjiang/gradio/样品1 LG 65UF8580/华为P50手机采集图像/监视器采集图像/1.jpg')
print("Number of Keypoints Detected:", len(keypoints))
if descriptors is not None:
    print("Descriptors Shape:", descriptors.shape)
else:
    print("No descriptors found.")