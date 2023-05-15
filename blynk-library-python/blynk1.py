"""
This code i Belogs to SME Dehradun Firm. For any query, mail us at schematicslab@gmail.com 
"""
import os
import BlynkLib
import RPi.GPIO as GPIO
from BlynkTimer import BlynkTimer

#define BLYNK_TEMPLATE_ID "TMPL3bFLs7oma"
#define BLYNK_TEMPLATE_NAME "PLR"
BLYNK_AUTH_TOKEN = "cTUHSyMJi9_db227gb1pEqN5x96JsHuZ"

#BLYNK_AUTH_TOKEN = 'JvbuHScbVW1XUW5CA4w9jI_zQ01UN-JF'

cmd = "python3 updated.py"

led1 = 18
led2 = 19
GPIO.setmode(GPIO.BCM)
GPIO.setup(led1, GPIO.OUT)
GPIO.setup(led2, GPIO.OUT)

x = 20
# Initialize Blynk
blynk = BlynkLib.Blynk(BLYNK_AUTH_TOKEN)

# Led control through V0 virtual pin
@blynk.on("V0")
def v0_write_handler(value):
#    global led_switch
    if int(value[0]) is not 0:
        #GPIO.output(led1, GPIO.HIGH)
        print('LED1 HIGH')
        os.system(cmd)
    else:
        #GPIO.output(led1, GPIO.LOW)
        print('LED1 LOW')

# Led control through V0 virtual pin
@blynk.on("V1")
def v1_write_handler(value):
#    global led_switch
    if int(value[0]) is not 0:
        GPIO.output(led2, GPIO.HIGH)
        print('LED2 HIGH')
    else:
        GPIO.output(led2, GPIO.LOW)
        print('LED2 LOW')

#function to sync the data from virtual pins
@blynk.on("connected")
def blynk_connected():
    print("Raspberry Pi Connected to New Blynk") 

while True:
    blynk.run()
