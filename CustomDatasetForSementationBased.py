from PIL import Image
import torch
import cv2
import string
import numpy as np

# Parameters:
NUMBER = ["{}".format(x) for x in range(10)]
ALPHABET = list(string.ascii_uppercase)
TABLE = NUMBER+ALPHABET # The table for CAPTCHA
LEN_OF_TABLE = len(TABLE) # in total 10+26 alphanumeric characters
#BATCH_SIZE = 100
LEN_OF_CAPTCHA = 6 # each picture contains 6 characters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=None, target_transform=None, height=50, width=200):
        self.transform = transform
        self.target_transform = target_transform
        self.images = images
        self.width  = width
        self.height = height

    def __getitem__(self, index):
        # Get the image with path
        image = cv2.imread(str(self.images[index]))
        # Increase contrast: segmentation-based so the preprocessing is more complicated
        image = cv2.convertScaleAbs(image, alpha=3, beta=40)
        # Erode noise
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        # Convert the image into grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        # Resize the image to ensure the size
        image = cv2.resize(image, (self.width, self.height))
        # Binarization of images
        _, image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        # Method from dsp.stackexchange.com/questions/52089/removing-noisy-lines-from-image-opencv-python
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        # Shear transformation, from thepythoncode.com/article/image-transformations-using-opencv-in-python#Image_Shearing
        M = np.float32([[1, -0.5, 0],
                        [0,    1, 0],
                        [0,    0, 1]])
        rows, cols = image.shape #(50, 200)
        image = cv2.warpPerspective(image,M,(int(cols),int(rows)), cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        # Horizontal stretch
        M = np.float32([[1.2, 0, 0],
                        [0,   1, 0],
                        [0,   0, 1]])
        rows, cols = image.shape #(50, 200)
        image = cv2.warpPerspective(image,M,(int(cols),int(rows)), cv2.INTER_LINEAR, borderValue=(255, 255, 255))

        label = captcha_to_vector(self.images[index].name.split("_")[0])
        img_seg_list = []
        label_lst = []
        # Segmentation 
        for j in range(LEN_OF_CAPTCHA):
            left = (j+1)*25
            right = (j+2)*25
            im_seg = image[:, left:right]
            # Apply the transform to the image
            if self.transform is not None:
                img_seg_list.append(self.transform(im_seg))
            else:
                img_seg_list.append(im_seg)
            label_lst.append(label[j*36:(j+1)*36])
        return img_seg_list, label_lst

    def __len__(self):
        return len(self.images)

# Convert the CAPTCHA into the (6*36,) vector (6 characters, 10 numbers + 26 uppercase/capital characters)
# 1 means the CAPTCHA image contains this character in TABLE, 0 means otherwise
def captcha_to_vector(captcha_str):
    captcha_str = captcha_str.upper()
    vector = np.zeros(36*6, dtype=np.float32)
    for i, char in enumerate(captcha_str):
        ascii = ord(char)
        if 48 <= ascii <= 57:
            index = ascii-48
        elif 65 <= ascii <= 90:
            index = ascii-ord('A')+10
        vector[i*36+index] = 1.0
    return vector

# Convert the vector to the CAPTCHA (the input vector is different from the vector above)
# Example: input: [1,2,34,2,6,7]; output: "23Y378"
def vector_to_captcha(vector):
    captcha_str = ""
    for i in vector:
        captcha_str += TABLE[i]
    return captcha_str

def get_data(dataloader):
    X = []
    Y = []
    n = len(dataloader)
    print("")
    for z in range(n):
        if (z+1)%100 == 0: print("{}".format(z+1))
        i, l = next(iter(dataloader))
        for image, label in zip(i, l):
            image = image.to(device)
            label = label.to(device)
            label = label.reshape(1, 36)
            label = torch.argmax(label, dim=1)
            label = vector_to_captcha(label)
            image = image.reshape(image.shape[2], image.shape[3]).cpu()
            X.append(image.flatten().tolist())
            Y.append(label)
    new_Y = []
    for j in range(len(Y)):
        new_Y.append(Y[j][0])
    return X, Y

# Groups the original and the predicted characters together to into CAPTCHAs
def group(lst):
    n = len(lst)
    i = 0
    new_list = []
    while i < n:
        captcha = lst[i:i+5] # six per group
        new_list.append(''.join(captcha))
        i += 6
    return new_list