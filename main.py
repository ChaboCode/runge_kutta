from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import matplotlib.pyplot as plt
from math import *
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root_page():
    return {"Hola": "sexo"}


@app.get("/values")
def grunt_kutta_route(f_str: str, x0: float, y0: float, steps: int, end: float):
    h = (end - x0) / steps
    f = eval(f"lambda x, y: {f_str}")
    result, y, k1, k2, k3, k4 = runge_kutta(f, x0, y0, h, end)
    return {
        "result": result,
        "steps": y,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "k4": k4,
        "xn": list(np.arange(x0, end +h, h)),
    }


@app.get("/img")
def runge_kutta_pic(f_str: str, x0: float, y0: float, steps: int, end: float):

    h = (end - x0) / steps
    f = eval(f"lambda x, y: {f_str}")

    result, y, k1, k2, k3, k4 = runge_kutta(f, x0, y0, h, end)
    x = np.arange(x0, end + h, h)

    fig, ax = plt.subplots()
    ax.plot(x, y, "o")

    f_poly, x_poly, y_poly = approximate(x, y)
    print(f_poly)
    ax.plot(x_poly, y_poly, "-")
    ax.grid()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")


def runge_kutta(f, x0, y0, h, end):
    y = [y0]
    yn = y0
    k1s = []
    k2s = []
    k3s = []
    k4s = []
    for i in np.arange(x0, end, h):
        xn = i
        k1 = f(xn, yn)
        k1s.append(k1)
        # print('k1: ', k1)

        k2 = f(xn + h / 2, yn + (h / 2) * k1)
        k2s.append(k2)
        # print('k2: ', k2)

        k3 = f(xn + h / 2, yn + (h / 2) * k2)
        k3s.append(k3)
        # print('k3: ', k3)

        k4 = f(xn + h, yn + h * k3)
        k4s.append(k4)
        # print('k4: ', k4)

        yn = yn + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y.append(yn)
        # print('yn: ', yn)
    return yn, y, k1s, k2s, k3s, k4s


def approximate(x, y):
    coef = np.polyfit(x, y, deg=3)
    f_poly = np.poly1d(coef)
    x_poly = np.linspace(min(x), max(x), 100)
    y_poly = f_poly(x_poly)

    return f_poly, x_poly, y_poly
