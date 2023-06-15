import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
from sklearn.preprocessing import StandardScaler

with open("image1.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
st.markdown(
f"""
<style>
.stApp {{
    background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
    background-size: cover
}}
</style>
""",
unsafe_allow_html=True
)

st.title("Check your Price")

st.markdown("Tell us about your preference ")

#load pickle files

with open ("XGB_model.pkl","rb")as file:
    xgb = pickle.load(file)
    
with open ("Scaling_attraction_index.pkl","rb") as file:
    att_i = pickle.load(file)

with open ("Scaling_attraction_index_norm.pkl","rb") as file:
    att_n = pickle.load(file)
    
with open ("Scaling_city_center_dist.pkl","rb") as file:
    city_d = pickle.load(file)

with open ("city_encoding.pkl","rb") as file:
    city_enc = pickle.load(file)
    
with open ("host_type_encoding.pkl","rb") as file:
    host_enc = pickle.load(file)

with open ("Scaling_metro_dist.pkl","rb") as file:
    me_d = pickle.load(file)
    
with open ("price.pkl","rb") as file:
    pri = pickle.load(file)   
    
with open ("Scaling_restaurant_index.pkl","rb") as file:
    res_i = pickle.load(file)   

with open ("Scaling_restaurant_index_norm.pkl","rb") as file:
    res_n = pickle.load(file) 
    
with open ("room_type_encoding.pkl","rb") as file:
    room_t = pickle.load(file) 
    
with open ("Scaling_person_capacity.pkl","rb") as file:
    pers = pickle.load(file) 
    
with open ("Scaling_cleanliness_rating.pkl","rb") as file:
    clean1 = pickle.load(file)    
    
with open ("Scaling_guest_satisfaction_overall.pkl","rb") as file:
    guest = pickle.load(file)    
    
with open ("Scaling_bedrooms.pkl","rb") as file:
    bedrooms = pickle.load(file) 
    
with open ("price.pkl","rb") as file:
    pri = pickle.load(file)
    
#get user input

bedroom = st.slider("Enter the no.of bedrooms",0,6,1)
person = st.number_input("Enter the person capacity",2,6)
clean = st.number_input("Expected cleanliness rating",2,10)
guest_rat = st.number_input("Expected guest rating",20,100)
city_dist = st.number_input("Expected distance from city",0,26)
metro_dist = st.number_input("Expected distance from metro station",0,16)
att_in = st.number_input("Expected attraction index",14,4514)
att_norm = st.number_input("Expected attraction index norm",0,100)
res_in = st.number_input("Expected restaurent index",19,6697)
res_norm = st.number_input("Expected restaurent index norm",0,100)
city_user = st.selectbox("Choose the city",(['Amsterdam', 'Athens', 'Barcelona', 'Berlin', 'Budapest', 'Libson',
       'London', 'Paris', 'Rome', 'Vienna']))
room = st.selectbox("Choose room type",(['Private room', 'Entire home/apt', 'Shared room']))
host = st.selectbox("Choose host type",(['Single host', 'Multiple host']))
superhost = st.selectbox("Is superhost",(["Yes","No"]))
day = st.selectbox("choose the day",(["Weekend","Weekday"]))


#preprocessing
bed = bedrooms.transform(pd.DataFrame([bedroom]))
per = pers.transform(pd.DataFrame([person]))
cle = clean1.transform(pd.DataFrame([clean]))
gue = guest.transform(pd.DataFrame([guest_rat]))
cit_d = city_d.transform(pd.DataFrame([city_dist]))
met = me_d.transform(pd.DataFrame([metro_dist]))
att = att_i.transform(pd.DataFrame([att_in]))
at_n = att_n.transform(pd.DataFrame([att_norm]))
res = res_i.transform(pd.DataFrame([res_in]))
re_n = res_n.transform(pd.DataFrame([res_norm]))
cit = city_enc.transform(pd.DataFrame({"city":[city_user]}))
ro = room_t.transform(pd.DataFrame({"room_type":[room]}))
ho = host_enc.transform(pd.DataFrame({"host_type":[host]}))
su = [1 if superhost=="Yes" else 0 ]
da = [1 if day=="Weekend" else 1]

#user i/p to model i/p

data = {"city" : cit.iloc[0,0] ,
        "day" : da[0],
         "room_type":ro.iloc[0,0],
         "person_capacity" : per[0],
         "is_superhost" : su[0],
          "cleanliness_rating" : cle[0],
          "guest_satisfaction_overall" : gue[0],
          "bedrooms" :bed[0],
           "city_center_dist" : cit_d[0],
            "metro_dist" : met[0],
             "attraction_index" : att[0],
             "attraction_index_norm" : at_n[0],
              "restaurant_index" : res[0],
               "restaurant_index_norm" : re_n[0],
               "host_type":ho.iloc[0,0]}
data = pd.DataFrame([data]).apply(pd.to_numeric,errors='coerce')

#st.write(type(per[0]))

prediction = xgb.predict(data)
prediction = pri.inverse_transform([prediction])
if st.button("Know your price"):
    st.success(f"the price is {round(prediction[0,0],2)}")
    
#invers
