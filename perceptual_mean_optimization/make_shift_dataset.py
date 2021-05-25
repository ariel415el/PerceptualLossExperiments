import os
import cv2
import numpy as np
root = "image_clusters"
for edge_size in [128,64,32,16]:
    for size in [64]:
        for path in os.listdir(root):
            if path.endswith(".png"):
                img = cv2.imread(os.path.join(root, path))
                dir_name = os.path.join(root, f"{os.path.splitext(path)[0]}_s{size}_c_{edge_size}", 'images')
                os.makedirs(dir_name, exist_ok=True)
                for i in range(32):
                    x1 = np.random.randint(0, edge_size)
                    y1 = np.random.randint(0, edge_size)
                    img2 = img[y1:img.shape[0]-edge_size+y1]
                    img2 = img2[:, x1:img.shape[1]-edge_size+x1]
                    img2 = cv2.resize(img2, (size, size))
                    x = cv2.imwrite(os.path.join(dir_name, f"img_{i}.png"), img2)
                    print(x)