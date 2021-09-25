import requests
from mylib.config import five_mins, people_change
import datetime
  
# defining the api-endpoint 
#http://192.168.13.250:7001/api/createEvent?source=CFA&caption=%E4%BA%BA%E5%93%A1%E9%80%B2%E5%87%BA&description=enter:100%0D%0Aexit:95%0D%0Atotal:5
#API_ENDPOINT = "http://admin:hdit23059797@192.168.13.250:7001/api/createEvent?source=CFA&caption=people_counting&description=enter:3\r\nexit:4"
#http://192.168.13.250:7001/api/createEvent?source=CFA&caption=%E4%BA%BA%E5%93%A1%E9%80%B2%E5%87%BA&description=enter:10%0D%0Aexit:5

#API_ENDPOINT = "http://192.168.13.250:7001/api/createEvent/"
def send_req(enter,exit):
        API_ENDPOINT = "http://admin:hdit23059797@192.168.13.250:7001/api/createEvent/"
        #http://admin:hdit23059797@192.16813.250:7001/api/createEvent
        # your API key here
        #API_KEY = "admin:hdit23059797"
        #?source=CFA&caption=%E4%BA%BA%E5%93%A1%E9%80%B2%E5%87%BA&description=enter:100%0D%0Aexit:95%0D%0Atotal:5
        #now = datetime.datetime.now()
        # data to be sent to api
        total=enter-exit
        data = {
                'source':'CFA',
                'caption':'%E4%BA%BA%E5%93%A1%E9%80%B2%E5%87%BA',
                'description':f"enter={enter}%0D%0Aexit={exit}%0D%0Atotal={total}"
                }
        #API_ENDPOINT = 'http://192.168.13.250:7001/api/createEvent?source=CFA&caption=%E4%BA%BA%E5%93%A1%E9%80%B2%E5%87%BA&description=enter:100%0D%0Aexit:95%0D%0Atotal:5'
        #API_ENDPOINT = "http://192.168.13.250:7001/api/createEvent?"+f"source={data['source']}"+f"&caption={data['caption']}"+f"&description={data['description']}"
        # sending post request and saving response as response object
        
        #r = requests.post(url = API_ENDPOINT)


        
        # extracting response text
        #print('success')
        #pastebin_url = r.text
        #print("The pastebin URL is:%s"%pastebin_url)
        #print("status "+str(r.status_code))
        requests.get(url = API_ENDPOINT, params=data)
