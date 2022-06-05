from argparse import FileType
import requests
import base64
import cv2

def _tobase64(imgurl):

    img = cv2.imread(imgurl)
    jpg_img = cv2.imencode('.jpg', img)
    b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')
    return(b64_string)

sampleImage = './images/26187.png'

url = 'https://api.ocr.space/parse/image'
myobj = {'apikey': 'K83405027688957','url': sampleImage, 'language': 'eng', 'filetype': 'png'}

x = requests.post(url, data = myobj, headers= {'content_type': 'image/jpg'})

print(x.text)
'data:image/png'