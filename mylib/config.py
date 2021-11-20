import datetime
#===============================================================================
""" Optional features config. """
#===============================================================================

# Enter your email below. This email will be used to send alerts.
# E.g., "email@gmail.com"
# Enter the email password below. Note that the password varies if you have secured
# 2 step verification turned on. You can refer the links below and create an application specific password.
# Google mail has a guide here: https://myaccount.google.com/lesssecureapps
# For 2 step verified accounts: https://support.google.com/accounts/answer/185833
sender_email='testsenderemail35@gmail.com'
sender_email_password='testsenderemailpassword'

# Enter mail below to receive real-time email alerts
# e.g.,s 'email@gmail.com'
MAIL = 'winwongsawatdichart@gmail.com'
# Enter the ip camera url (e.g., url = 'http://191.138.0.100:8040/video')
url = "rtsp://admin:123456@192.168.1.110/onvif-media/media.amp?streamprofile=Profile1&audio=0"
#rtsp://admin:Hdit23059797@192.168.13.125:554/ISAPI/Streaming/channels/101
# ON/OFF for mail feature. Enter True to turn on the email alert feature.
ALERT = True
# Set max. people inside limit. Optimise number below: 10, 50, 100, etc.
Threshold = 100
# Threading ON/OFF
Thread = True

############ detection line modification parameters ##################
# height and width of frame
# start running the video first, it's height and width will be printed in the cmd then type it in here
# hi = height and wi = width
hi=500
wi=500

#specify points here the line will be draw from x1,y1 to x2,y2
#note that when hi = 0, it will be at the top and when hi = max it will be at the bottom
x1=0
y1=hi  * 0.5
x2=wi
y2=hi  * 0.5

# set vertical_direction to 1 for catagorize people in a vertical movement
# set to 0 for catagorize people by horizon movement
# if the line has a negative or positive slope, set to 0 or 1 will do
#0 will use movement up/downward for calculations, 1 will use movement left/right for calculations
vertical_direction = 1
#Specify enter side 
# if vertical_direction = 1 , use "up" for enter up and "down" for enter down
# if vertical_direction = 0 , use "left" for enter left side and "right" for enter right side
enter_direction = 'up'
############################
# Simple log to log the counting data
Log = True
# choose the time to update Log below, 
# for example if we want to update every 1 hour it would be write as timedel= datetime.timedelta(seconds=0,minutes=0,hours=1,days=0,weeks=0)
timedel= datetime.timedelta(seconds=0,minutes=0,hours=1,days=0,weeks=0)
# camera place name
cam_place='1st floor'
####################
# Auto run/Schedule the software to run at your desired time
Scheduler = True
# Auto stop the software after certain a time/hours
Timer = False
#requests config
five_mins= True
people_change= True
# request API end-point in this format for example : 'http://admin:hdit23059797@192.168.1.11:7001/api/createEvent?'
request_API='http://admin:hdit23059797@192.168.1.11:7001/api/createEvent?'
#===============================================================================
#===============================================================================
