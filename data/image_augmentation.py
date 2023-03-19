import random
import numpy as np
import cv2
import glob
import os


# peopleDir = './images/people_crops/'

class ImageAugmentation():
    def __init__(self, image_dir, save_dir, people_dir, occlusion_level="low"):
        self.batchSize = 1
        self.crop_size = [224, 224]
        self.image_size = [256, 256]
        self.image_files = glob.glob(os.path.join(image_dir, '*.jpg'))
        self.people_crop_files = glob.glob(os.path.join(people_dir,'*.png'))
        self.save_dir = os.path.join(save_dir, occlusion_level)
        self.occlusion_level = occlusion_level
    
    def processAllImages(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        for ix in range(0, len(self.image_files)):
            img = self.getProcessedImage(self.image_files[ix])
            try:
                cv2.imwrite(os.path.join(self.save_dir, self.image_files[ix].split("/")[-1]), img[0])
            except:
                print("cannot save {}".format(im))
        pass

    def getProcessedImage(self, image_file):
        img = cv2.imread(image_file)
        if img is None:
            return None
        img = cv2.resize(img, (self.image_size[0], self.image_size[1]))
        top = np.random.randint(self.image_size[0] - self.crop_size[0])
        left = np.random.randint(self.image_size[1] - self.crop_size[1])
        img = img[top:(top+self.crop_size[0]),left:(left+self.crop_size[1]),:]
        return self.getPeopleMasks(img)

        pass

    def getPeopleMasks(self, input_image):
        # input_image_resized = cv2.resize(input_image, tuple(self.crop_size[::-1]))
        which_inds = random.sample(list(np.arange(0,len(self.people_crop_files))),self.batchSize)

        people_crops = np.zeros([self.batchSize,self.crop_size[0],self.crop_size[1]])
        for ix in range(0,self.batchSize):
            people_crops[ix,:,:] = self.getImageAsMask(self.people_crop_files[which_inds[ix]])

        people_crops = np.expand_dims(people_crops, axis=3)

        # apply mask to input image
        masked_image = input_image * (1 - people_crops)

        return masked_image

    def getImageAsMask(self, image_file):
        img = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # how much of the image should the mask take up
        if self.occlusion_level == "random":
            scale = np.random.randint(30,70)/float(100)
        elif self.occlusion_level == "low":
            scale = 0.30
        elif self.occlusion_level == "medium":
            scale = 0.50
        else:
            scale = 0.70
        resized_img = cv2.resize(img,(int(self.crop_size[0]*scale),int(self.crop_size[1]*scale)))

        # where should we put the mask?
        top = np.random.randint(0,self.crop_size[0]-resized_img.shape[0])
        left = np.random.randint(0,self.crop_size[1]-resized_img.shape[1])

        new_img = np.ones((self.crop_size[0],self.crop_size[1]))*255.0
        new_img[top:top+resized_img.shape[0],left:left+resized_img.shape[1]] = resized_img

        new_img[new_img<255] = 1
        new_img[new_img>1] = 0

        return new_img
    
if __name__ == "__main__":
    image_dir = "./images/train/"
    people_dir = './images/people_crops/'
    '''
    image_dir: where all the downloaded images are
    save_dir: prcessed images will be saves in save_dir/occlusion_level
    occlusion_level: low, medium, high, random
    '''
    im = ImageAugmentation(image_dir, save_dir="./images/processed", people_dir=people_dir, occlusion_level="low")
    im.processAllImages()
