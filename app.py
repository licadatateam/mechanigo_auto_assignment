# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import signal
import sys
from types import FrameType

from flask import Flask, jsonify, request

from utils.logging import logger
import MechanicsAssignment as mech_assign
import chromedriver_binary
import utils_mgo
import json
import datetime as dt

app = Flask(__name__)


@app.route("/")
def hello() -> str:
    # Use basic logging with custom fields
    logger.info(logField="custom-entry", arbitraryField="custom-entry")

    # https://cloud.google.com/run/docs/logging#correlate-logs
    logger.info("Child logger with trace Id.")

    return "Hello, World!"

@app.route('/assignment/', methods = ['GET'])
def get_assignments():
    '''
    Sample usage: requests.get('http://127.0.0.1:5000/assignment/?date=2024-10-19')
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

def shutdown_handler(signal_int: int, frame: FrameType) -> None:
    logger.info(f"Caught Signal {signal.strsignal(signal_int)}")

    from utils.logging import flush

    flush()

    # Safely exit program
    sys.exit(0)


if __name__ == "__main__":
    # Running application locally, outside of a Google Cloud Environment

    # handles Ctrl-C termination
    signal.signal(signal.SIGINT, shutdown_handler)

    app.run(host="localhost", port=8080, debug=True)
else:
    # handles Cloud Run container termination
    signal.signal(signal.SIGTERM, shutdown_handler)
