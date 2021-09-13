from mylib.mailer import Mailer
from mylib import config
import cv2

def check_exceed(x,frame):
    if sum(x) >= config.Threshold:
        cv2.putText(frame, "-ALERT: People limit exceeded-", (10, frame.shape[0] - 80),
        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
        if config.ALERT:
            print("[INFO] Sending email alert..")
            Mailer().send(config.MAIL)
            print("[INFO] Alert sent")