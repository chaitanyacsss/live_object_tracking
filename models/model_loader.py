from torchvision import transforms

from models import *
from utils import non_max_suppression, load_classes


class ModelLoader(object):
    def __init__(self, imgsize, weightspath, classpath, confpath, confthres, nmsthres):
        self.imgsize = imgsize
        self.weightspath = weightspath
        self.classpath = classpath
        self.confpath = confpath
        self.confthres = confthres
        self.model = Darknet(self.confpath, img_size=self.imgsize)
        self.model.load_weights(weightspath)
        self.model.cuda()
        self.model.eval()
        self.classes = load_classes(classpath)
        self.nmsthres = nmsthres

    def get_model(self):
        return self.model

    def get_classes(self):
        return self.classes

    def get_imgsize(self):
        return self.imgsize

    def get_transforms(self, imagew, imageh):
        return transforms.Compose([transforms.Resize((imageh, imagew)), transforms.Pad(
            (max(int((imageh - imagew) / 2), 0), max(int((imagew - imageh) / 2), 0),
             max(int((imageh - imagew) / 2), 0), max(int((imagew - imageh) / 2), 0)),
            (128, 128, 128)), transforms.ToTensor(), ])

    def detect_from_image(self, img):
        img_size = self.imgsize
        ratio = min(img_size / img.size[0], img_size / img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        image_tensor = self.get_transforms(imw, imh)(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img = Variable(image_tensor.type(torch.cuda.FloatTensor))
        with torch.no_grad():
            detections = self.model(input_img)
            detections = non_max_suppression(detections, 80, self.confthres, self.nmsthres)
        return detections[0]
