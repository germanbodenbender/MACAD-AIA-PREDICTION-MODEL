#AIA-COMPUTER VISION SEMINAR
#STUDENT: German Otto Bodenbender
#Testing using Hops to automate the preiction process on this Pattern Model example 

from typing import Literal
from flask import Flask
import ghhops_server as hs
from ghhops_server.params import HopsParamAccess
import rhino3dm
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


app = Flask(__name__)
hops = hs.Hops(app)

@hops.component(
    "/runml",
    name="runml",
    description="Run Machine Learning prediction",
    icon="examples/pointat.png",
    inputs=[
        hs.HopsBoolean("B","BO","Boolean"),
    ],
    outputs=[
        hs.HopsPoint("P", "P", "Point on curve at t"),
    ]
)


def run_ml():
    if hs.HopsBoolean == True:
        exec('runPython.py')
        print("Model OK")


if __name__ == "__main__":
    app.run()