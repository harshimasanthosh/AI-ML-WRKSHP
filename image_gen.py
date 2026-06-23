import cv2
import numpy as np

image = cv2.imread("image.png.png")

grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

small_image = cv2.resize(grey_image, (600, 600))

sobel_x = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
)

sobel_y = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]
)

output1 = cv2.filter2D(small_image, -1, sobel_x)
output2 = cv2.filter2D(small_image, -1, sobel_y)

sobel_combined = cv2.magnitude(output1.astype(np.float32), output2.astype(np.float32))

print(sobel_combined.dtype, sobel_combined.max())

cv2.imwrite("edges.png", cv2.convertScaleAbs(sobel_combined))
cv2.imshow("greyscale image", small_image)
cv2.imshow("combined", cv2.convertScaleAbs(sobel_combined))
cv2.imshow("Sobel-X Output", output1)
cv2.imshow("Sobel-Y Output", output2)

cv2.imwrite("Grayscale Image.jpg", small_image)
cv2.imwrite("combined.jgp", cv2.convertScaleAbs(sobel_combined))
cv2.imwrite("Sobel-X Output.jpg", output1)
cv2.imwrite("Sobel-Y Output.jpg", output2)

cv2.waitKey(0)
cv2.destroyAllWindows()