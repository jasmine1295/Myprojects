from typing import Union
from fastapi import FastAPI ,Request
from pydantic import BaseModel
import json
import imdrf_1
import datetime
import requests
from imdrf_1 import get_anne_data, code_list
import yaml
from pathlib import Path
import sqlite3
import json

class Item(BaseModel):
    text: str
    includeMeddra: int
    itemReturned : int



script_dir = str(Path( __file__ ).parent.absolute())
with open(script_dir+"/"+'config.yml', 'r') as file:
    all_config = yaml.safe_load(file)
directory = all_config["Path_dtl"]

def check_apikey(key):
    url_path = directory["URL"]
    url = url_path
    #url = "https://imdrf-console.vc-19.ai/api/v1/check_key"
    #url = "https://imdrf-qaadmin.vc-19.ai/api/v1/check_key"
    payload = ""
    headers = {

      'check-key': key
      #'O8Ao5D2EDtWY6kPoHf0WvMTbCzvUUG46zMaL9oDEAQMADQSd1RiLXXEJlgskQFm5LyPmoi6BGJDVqFMpi9cjwKU8QR0tDFTKjzDt96rJx7I9oRsy1ny8jUu-OhOHnn4bzPQGTw10692c40-4735-40b2-8b91-ebb2bd33f333'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    #print(response)
    return response

print("success")

app = FastAPI()

#### Below code added for browser compatability on 3rd Feb 2023
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
#app = FastAPI()
origins = ["http://localhost:3000/",
           "https://3a-pvsafety-dev.vc-19.ai/",
           "https://3a-pvsafety-qa.vc-19.ai/",
           "https://exelixisdemo.vc-19.ai/",
           "https://pmi.vc-19.ai/",
           "https://drugsdemo.vc-19.ai/"]
app.add_middleware(CORSMiddleware,
                   allow_origins= ["*"], #origins
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   
                   )
#### Above code added for browser compatability on 3rd Feb 2023


@app.get("/bausch/version")
def get_version():
      #current_time = datetime.datetime.now()
      return {"version":directory["VERSION"]}
    
@app.post("/bausch/api/v1/imdrf")
def create_item(item: Item, request: Request):
  #if 'succuss' in json.loads(check_apikey(request.headers.get('check-key')).text)['message']:
  try:
    if (directory["api_str"] in json.loads(check_apikey(request.headers["Authorization"].split()[1]).text)['message']) or (directory["api_str"] in json.loads(check_apikey(request.headers.get('check-key')).text)['message']):

        try:

            if len(item.text.split()) != 0:   # Added for if foi text is empty 
                if len(set(item.text.lower().split()) & set(code_list)) == 0:  # added for if foi text contains any annex code  
                    a_pred = imdrf_1.prediction_A([item.text],6,directory["a_pred"])
                    b_pred = imdrf_1.prediction_B([item.text],6,item.itemReturned,directory["b_pred"])
                    c_pred = imdrf_1.prediction_C([item.text],6,directory["c_pred"])
                    d_pred = imdrf_1.prediction_D([item.text],6,directory["d_pred"])
                    e_pred = imdrf_1.prediction_E([item.text],6,item.includeMeddra,directory["e_pred"])
                    f_pred = imdrf_1.prediction_F([item.text],6,directory["f_pred"])
                    g_pred = imdrf_1.prediction_G([item.text],6,item.itemReturned,directory["g_pred"])

                else:
                    fo_text = list(set(item.text.lower().split()) & set(code_list))[0]
                    return {"result":{"Annex_"+str(fo_text[0:1]).upper()+"_Predictions":get_anne_data(fo_text,item.includeMeddra,item.text).to_dict(orient='records')}}
            else:
                return {"error":"Unable to complete request 1"}
        except Exception as e:
            print(e)
            return {"error":"Unable to complete request 2"}

        #out_json = {"result":{"Annex_A_Predictions":a_pred.to_dict(orient='records'),"Annex_E_Predictions":e_pred.to_dict(orient='records'),"Annex_F_Predictions":f_pred.to_dict(orient='records')}}
        out_json = {"result":{"Annex_A_Predictions":a_pred.to_dict(orient='records'),"Annex_B_Predictions":b_pred.to_dict(orient='records'),"Annex_C_Predictions":c_pred.to_dict(orient='records'),"Annex_D_Predictions":d_pred.to_dict(orient='records'),"Annex_E_Predictions":e_pred.to_dict(orient='records'),"Annex_F_Predictions":f_pred.to_dict(orient='records'),"Annex_G_Predictions":g_pred.to_dict(orient='records')}}  #,"Annex_G_Predictions":g_pred.to_dict(orient='records')
        
        conn = sqlite3.connect('imdrf.db')
        cursor = conn.cursor()
        # with open('out_json.json', 'r') as f:
        #     data = f.read()
        #json_data = json.loads(out_json)

        out_json_1 = str(out_json)
        #print(out_json)
        
        
        cursor.execute("INSERT INTO API_OUT (Content) VALUES (?)", (out_json_1,))
        conn.commit()
        conn.close()
        
        return out_json
    else:
        return {'error':'Invalid key'} 
  except:
    print('r3')
    return {"error":"Unable to complete request"}
