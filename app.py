# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:22:45 2024

@author: carlo
"""

from flask import Flask, jsonify, request
import json
import datetime as dt
# custom files
import MechanicsAssignment as mech_assign
import utils_mgo

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route('/assignment/', methods = ['GET'])
def get_assignments():
    '''
    Sample usage: r = requests.get('http://127.0.0.1:5000/assignment/?date=2024-10-19')
    
    Sample parse: pd.DataFrame(json.loads(r.content)['data'])
    '''
    # extract variables/parameters
    selected_date = request.args.get('date')
    # if selected_date is provided
    if selected_date is not None:
        try:
            selected_date, selected_sched = utils_mgo.check_selected_date(selected_date)
            # set selected_date in assignment class
            assignment = mech_assign.MechanicsAssignment(selected_date)
            # calculate solution
            assignment.optimize_assignment()
            # return result in json form
            results = assignment.app_result
            response = jsonify(data=json.loads(results))
            
        except Exception as exception:
            raise exception
    
    return response

if __name__ == "__main__":
    app.run(port = 5000)
       
    
        
    