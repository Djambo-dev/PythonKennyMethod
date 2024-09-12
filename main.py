import numpy as np
from scipy.ndimage import convolve
import cv2
import matplotlib.pyplot as plt


def my_canny(image, low_threshold, high_threshold, kernel_size=3, sigma=1):
    # Step 1: Gaussian blur
    def gaussian_kernel(size, sigma):
        kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            -((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)), (size, size))
        return kernel / np.sum(kernel)

    def gaussian_blur(image, kernel_size, sigma):
        kernel = gaussian_kernel(kernel_size, sigma)
        return convolve(image.astype(np.float64), kernel)

    smoothed_image = gaussian_blur(image, kernel_size, sigma)

    # Step 2: Compute gradients (Sobel operators)
    def sobel_filters(image):
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        grad_x = convolve(image, sobel_x)
        grad_y = convolve(image, sobel_y)

        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)

        return grad_magnitude, grad_direction

    grad_magnitude, grad_direction = sobel_filters(smoothed_image)

    # Step 3: Non-maximum suppression
    def non_max_suppression(grad_magnitude, grad_direction):
        suppressed = np.zeros(grad_magnitude.shape, dtype=np.float64)
        for i in range(1, grad_magnitude.shape[0] - 1):
            for j in range(1, grad_magnitude.shape[1] - 1):
                direction = grad_direction[i, j]
                mag = grad_magnitude[i, j]

                # Determine neighboring pixels in the gradient direction
                if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                    before, after = grad_magnitude[i, j - 1], grad_magnitude[i, j + 1]
                elif 22.5 <= direction < 67.5:
                    before, after = grad_magnitude[i - 1, j - 1], grad_magnitude[i + 1, j + 1]
                elif 67.5 <= direction < 112.5:
                    before, after = grad_magnitude[i - 1, j], grad_magnitude[i + 1, j]
                else:
                    before, after = grad_magnitude[i - 1, j + 1], grad_magnitude[i + 1, j - 1]

                # Compare neighboring pixels and keep only local maxima
                if mag >= before and mag >= after:
                    suppressed[i, j] = mag

        return suppressed

    suppressed_image = non_max_suppression(grad_magnitude, grad_direction)

    # Step 4: Double thresholding and edge tracking by hysteresis
    def apply_threshold(image, low_threshold, high_threshold):
        strong_edges = (image >= high_threshold)
        weak_edges = (image < high_threshold) & (image >= low_threshold)
        return strong_edges.astype(np.uint8) * 255, weak_edges.astype(np.uint8) * 255

    low_threshold = int(low_threshold)
    high_threshold = int(high_threshold)

    strong_edges, weak_edges = apply_threshold(suppressed_image, low_threshold, high_threshold)

    # Step 5: Edge tracking by hysteresis
    def hysteresis_thresholding(strong_edges, weak_edges):
        h, w = strong_edges.shape
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if weak_edges[i, j]:
                    if np.any(strong_edges[i - 1:i + 2, j - 1:j + 2]):
                        strong_edges[i, j] = 255
                    else:
                        weak_edges[i, j] = 0

        return strong_edges

    edges_image = hysteresis_thresholding(strong_edges, weak_edges)

    return edges_image.astype(np.uint8)


# Load an image
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)

# Parameters for Canny edge detection
low_threshold = 50
high_threshold = 100

# Apply your custom Canny method
edges_custom = my_canny(image, low_threshold, high_threshold)

# Apply OpenCV's Canny method for comparison
edges_opencv = cv2.Canny(image, low_threshold, high_threshold)

# Save custom Canny result
cv2.imwrite('custom_canny_output.jpg', edges_custom)

# Save OpenCV Canny result
cv2.imwrite('opencv_canny_output.jpg', edges_opencv)

# Display results using matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(edges_custom, cmap='gray')
plt.title('Custom Canny')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(edges_opencv, cmap='gray')
plt.title('OpenCV Canny')
plt.axis('off')

plt.tight_layout()
plt.show()
