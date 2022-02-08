from flask import Flask, render_template,request
import numpy as np
import pandas as pd
from joblib import load
import plotly.express as px
import plotly.graph_objects as go
import uuid
 
app = Flask(__name__)

@app.route("/",methods=['GET','POST'])
def hello_world():
    request_type_str = request.method
    
    if request_type_str == 'GET':
        return render_template('index.html',href='static/panda.png')
    else:
        text = request.form["text"]
        np_arr = floats_string_to_np_arr(text)
        random_string = uuid.uuid4().hex
        path = "static/"+ random_string +".svg"
        model = load('model.joblib')
        
        make_picture("AgesAndHeights.pkl",model, np_arr,path)
        return render_template('index.html',href=path)
    
def make_picture(training_data_filename, model, new_input_np_arr,output_file):
    """
    
    output_file: svg
    """
    data = pd.read_pickle(training_data_filename) #should be a pkl
    ages = data['Age']
    data =data[ages>0]
    ages = data['Age']
    heights = data['Height']
    
    x_new = np.array(list(range(19))).reshape(19,1)
    preds = model.predict(x_new)
    fig = px.scatter(x=ages, y=heights,title= 'Heigth vs Ages', labels={'x': 'Age (years)',
                                                                        'y': 'Height (inches)'})
    fig.add_trace(go.Scatter(x=x_new.reshape(19),y=preds,mode='lines',name='model'))
    
    new_preds = model.predict(new_input_np_arr)
    fig.add_trace(go.Scatter(x=new_input_np_arr.reshape(len(new_input_np_arr)),y=new_preds,name='New Outputs',mode='markers',marker=dict(color='purple',size=20,line=dict(color='purple',width=2))))
    fig.write_image(output_file,width=800)
    fig.show()
    
def is_float(s):
    try:
        float(s)
        return True
    except:
        return False

def floats_string_to_np_arr(floats_str):
    floats = [float(x) for x in floats_str.split(',') if is_float(x)]
    return np.array(floats).reshape(len(floats),1)