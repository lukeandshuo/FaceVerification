import time
start_global = time.time()
import cv2
import os
import numpy as np
np.set_printoptions(precision=2)
import openface

class FaceVerification():

    def __init__(self,isDebug = False):

        fileDir = os.path.dirname(os.path.realpath(__file__))
        print fileDir
        modelDir = os.path.join(fileDir, '..', 'models')
        dlibModelDir = os.path.join(modelDir, 'dlib')
        dlibFacePredictor = os.path.join(dlibModelDir,"shape_predictor_68_face_landmarks.dat")
        openfaceModelDir = os.path.join(modelDir, 'openface')
        networkModel = os.path.join(openfaceModelDir,'nn4.small2.v1.t7')
        self.imgDim = 96
        self.align = openface.AlignDlib(dlibFacePredictor)
        self.net = openface.TorchNeuralNet(networkModel,self.imgDim)
        self.Debug = isDebug
        print "init time:",time.time() - start_global

    def getRep(self,imgPath):

        if self.Debug:
            print("Processing {}.".format(imgPath))
        bgrImg = cv2.imread(imgPath)
        if bgrImg is None:
            raise Exception("Unable to load image: {}".format(imgPath))
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

        if self.Debug:
            print("  + Original size: {}".format(rgbImg.shape))

        start = time.time()
        bb = self.align.getLargestFaceBoundingBox(rgbImg)
        drawImg = np.copy(bgrImg)
        cv2.rectangle(drawImg,(bb.left(),bb.top()),(bb.right(),bb.bottom()),(0,255,0),2)
        # cv2.imshow("bb",drawImg)
        # cv2.waitKey(0)
        if bb is None:
            raise Exception("Unable to find a face: {}".format(imgPath))
        if self.Debug:
            print("  + Face detection took {} seconds.".format(time.time() - start))

        start = time.time()
        landmarks = self.align.findLandmarks(rgbImg,bb)
        for i in landmarks:
            cv2.circle(drawImg,i,2,(0,0,255),2)
        # cv2.imshow("result",drawImg)
        # cv2.waitKey(0)
        alignedFace = self.align.align(self.imgDim, rgbImg, bb,
                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))
        if self.Debug:
            print("  + Face alignment took {} seconds.".format(time.time() - start))

        start = time.time()
        rep = self.net.forward(alignedFace)
        if self.Debug:
            print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
            print("Representation:")
            print(rep)
            print("-----\n")
        return rep, drawImg

    def verification(self,img1,img2,ret= dict(),threshold=0.9,size=(256,256)):
        starttime = time.time()
        rep1, drawImg1 = self.getRep(img1)
        rep2, drawImg2 = self.getRep(img2)
        resizedImg1 = cv2.resize(drawImg1, size)
        resizedImg2 = cv2.resize(drawImg2, size)
        d = rep1-rep2
        l2d=np.dot(d,d)
        if self.Debug:
            print "l2 distance:",l2d
        Same = "Different"
        if l2d<threshold:
            Same = "Same"
        endtime = time.time()
        ret['result'] = (True,Same,"%.3f"%(endtime-starttime))
        ret['drawImg1'] = resizedImg1
        ret['drawImg2'] = resizedImg2
        return Same,resizedImg1,resizedImg2
if __name__ =="__main__":
    img1 = "1.jpg"
    img2 = "3.jpg"
    face = FaceVerification(isDebug=True)
    start_global = time.time()
    Same,drawImg1,drawImg2 = face.verification(img1,img2)
    print "total time:", time.time() - start_global
    print Same
    # if Same:
    #     print "Same person!!!"
    # else:
    #     print "they are different person!!!"
    drawImg = np.concatenate((drawImg1,drawImg2),axis=1)
    cv2.imshow("final", drawImg)
    cv2.waitKey(3000)

