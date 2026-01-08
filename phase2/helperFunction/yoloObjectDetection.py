import cv2
import numpy as np

class yoloObjectDetection:
    def __init__(self):
        self.yolo = None
        
    def obj_detect(self, img, task='classify', mode='yolo'):
        detectedMask,detectedName,detectedBbox = [],[],[]
        if mode == 'yolo':
            imgProcessed = self.yolo.pre_process(img)
            results = self.yolo.predict(imgProcessed)
            if results.masks is None:
                return detectedMask,detectedName,detectedBbox
            
            boxes=results.boxes.xywh.cpu().numpy().astype(int)
            for b in boxes:
                w=b[2]
                h=b[3]
                x=b[0]-w//2
                y=b[1]-h//2
                detectedBbox.append((x,y,w,h))
            
            masks=results.masks.data.cpu().numpy()
            for m in masks:
                detectedMask.append(m)
            
            ids = results.boxes.cls.cpu().numpy()
            for i in ids:
                detectedName.append(results.names[i])

            return detectedMask,detectedName,detectedBbox
        
    def find_distance(self,depth,detectedMask):
        """Calculate distance of objects based on their binary mask and depth image.

        Args:
            depth (ndarray): Depth image.
            detectedMask (list): List of binary masks of the detected objects.

        Returns:
            list: Distance to detected objects.

        """
        detectedDist = []
        # ================   SECTION C.1 - Distance Estimation   ================
        for mask in detectedMask:
            isolated_depth = mask*depth.squeeze()
            distance = np.median(isolated_depth[isolated_depth.nonzero()])
            detectedDist.append(distance)
        return detectedDist

    def annotate(self,img,detectedName,detectedBbox,detectedDist):
        """Add names and bounding boxes of detected objects to the input image.

        Args:
            img (ndarray): RGB image.
            detected_name (list): List of names of detected objects.
            detected_bbx (list): List of bounding boxes of detected objects.

        """
        for i,bbox in enumerate(detectedBbox):
            x,y,w,h = bbox
            id = detectedName[i]
            dist = detectedDist[i]
            if dist == 0: dist_msg= ''
            else: dist_msg = ' ('+str(round(dist,2))+' m)'
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 1)
            cv2.putText(img, id+dist_msg, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255,0,255), 2)