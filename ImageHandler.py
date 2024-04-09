import cv2
import os
import numpy as np

class ImageHandler:
    
    # Initializer for the class, setting the directory
    def __init__(self, image_dir):
        self.image_dir = image_dir
        
    # Loads the images, standardizes to (224, 224) is standard model for tensorflow
    def image_load(self, target_size=(224,224)):
        
        original_images = []
        resized_images = []
        
        for filename in os.listdir(self.image_dir):
            
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(self.image_dir, filename)
                image = cv2.imread(image_path)
                
                #regular sized images
                original_images.append(image)
                
                #resized for possible model training on smaller image
                image = cv2.resize(image, target_size) #sets to target size
                image = image/255.0 #this will normalize the pixel values ranges (0,1)
                resized_images.append(image)
        
        return np.array(original_images), np.array(resized_images)
    
    def resize_image(self, image, width, height):
        return cv2.resize(image, (width, height))
    
if __name__ == '__main__': #this is made to test the image sizing
    img_handler = ImageHandler('TestImages/')
    original_img, resize_img = img_handler.image_load()
    
    for i, (original_img, resize_img) in enumerate(zip(original_img, resize_img)):
        
        # Sizes window to the size of display
        cv2.namedWindow('Original Lane Image {}'.format(i+1), cv2.WINDOW_NORMAL)
        
        #Displays the image before resizing
        cv2.imshow('Original Lane Image {}'.format(i+1), original_img)
        
        #helps resize the window
        h, w, _ = original_img.shape
        cv2.resizeWindow('Original Lane Image {}'.format(i+1), w, h)

        #Displays the resized forms of the images
        cv2.imshow('Resized'.format(i+1), resize_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    