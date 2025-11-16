# PACKAGES
import cv2
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

# SPEED LIMIT ( in px/s )
# This is an estimate in pixels per second based on the video's perspective.
# You may need to tune this value.
limit = 150 

# CLASS & METHODS
class EuclideanDistTracker:

    # PARAMETERS
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
        self.s1 = np.zeros((1, 1000))
        self.s2 = np.zeros((1, 1000))
        self.s = np.zeros((1, 1000))
        self.f = np.zeros(1000)
        self.capf = np.zeros(1000)
        self.count = 0
        self.exceeded = 0
        self.ids_DATA = []
        self.spd_DATA = []

    # UPDATE SPEED RECORD
    def update(self, objects_rect):
        objects_bbs_ids = []

        # CENTER POINTS
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # CHECK IF OBJECT IS DETECTED
            same_object_detected = False

            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 70:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True

                    # START TIMER
                    if (y >= 410 and y <= 430):
                        self.s1[0, id] = time.time()

                    # STOP TIMER and FIND DIFFERENCE
                    if (y >= 235 and y <= 255):
                        self.s2[0, id] = time.time()
                        self.s[0, id] = self.s2[0, id] - self.s1[0, id]

                    # CAPTURE FLAG
                    if (y < 235):
                        self.f[id] = 1

            # NEW OBJECT DETECTION
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1
                self.s[0, self.id_count] = 0
                self.s1[0, self.id_count] = 0
                self.s2[0, self.id_count] = 0

        # ASSIGN NEW ID TO OBJ
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids

    # SPEED FUNCTION ( in px/s )
    def getsp(self, id):
        if (self.s[0, id] != 0):
            # Speed = Distance / Time
            # Distance is ~200 pixels in your setup (420 - 245)
            s = 200 / self.s[0, id]
        else:
            s = 0

        return int(s)

    # SAVE VEHICLE DATA (IN MEMORY)
    def capture(self, img, x, y, h, w, sp, id):
        # This function no longer saves images to disk.
        # It only saves the data to lists for the app.
        if(self.capf[id] == 0):
            self.capf[id] = 1
            self.f[id] = 0
            self.count += 1
            
            if(sp > limit):
                self.exceeded += 1
            
            self.ids_DATA.append((id))
            self.spd_DATA.append((sp))

    # STORE DATA
    def dataset(self):
        return self.ids_DATA, self.spd_DATA

    # DATA VISUALIZATION
    def datavis(self, id_lst, spd_lst):
        x = id_lst
        y = spd_lst
        valx = [str(i) for i in x]
              
        # --- Create Figure and Axes ---
        # This will be returned to Streamlit
        fig, ax = plt.subplots(figsize=(20, 5))
        
        style.use('dark_background')
        
        # --- Plot using 'ax' object ---
        ax.axhline(y = limit, color = 'r', linestyle = '-', linewidth=5)
        ax.bar(x, y, width=0.5, linewidth=3, edgecolor='yellow', color='blue', align='center')
        ax.set_xlabel('ID')
        ax.set_ylabel('SPEED (px/s)')
        ax.set_xticks(x)
        ax.set_xticklabels(valx)
        ax.legend(["Speed Limit"])
        ax.set_title('SPEED OF VEHICLES CROSSING ROAD\n')
        
        # --- Return the figure object ---
        # Do NOT use plt.show() or plt.savefig()
        return fig

    # SPEED LIMIT
    def limit(self):
        return limit

    # end() function stub
    def end(self):
        # This function is called by app.py, but it no longer needs
        # to write to a text file. We leave it here as 'pass'
        # to prevent any errors.
        pass
