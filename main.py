from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from math import *

app  = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=['*'],
  allow_headers=['*'],
)

@app.get("/")
def root_page():
    return {"Hola": "sexo"}

@app.get("/grunt-kutta")
def grunt_kutta_route(f_str: str, x0: float, y0: float, h: float, end: float):
    f = eval(f'lambda x, y: {f_str}')
    result, y = runge_kutta(f, x0, y0, h, end)
    return {"result": result, "steps": y}

def runge_kutta(f, x0, y0, h, end):
  y = [y0]
  yn = y0
  for i in np.arange(x0, end, h):
    xn = i
    k1 = f(xn, yn)
    # print('k1: ', k1)

    k2 = f(xn + h/2, yn + (h/2) * k1)
    # print('k2: ', k2)

    k3 = f(xn + h/2, yn + (h/2) * k2)
    # print('k3: ', k3)

    k4 = f(xn + h, yn + h * k3)
    # print('k4: ', k4)

    yn = yn + (h/6) *(k1 + 2 * k2 + 2 * k3 + k4)
    y.append(yn)
    # print('yn: ', yn)
  return yn, y

