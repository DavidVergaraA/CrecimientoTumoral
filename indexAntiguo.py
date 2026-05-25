import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Tasa max de crecimiento tumoral (dia-1)
a = 0.18
# Inverso capacidad de carga del tejido (celulas-1)
b = 2e-9
# Influx basal de efectoras al sitio tumoral (celulas/dia)
s = 1.3e4
# Tasa de acumulacion estimulada de efectoras (día-1)       
p = 0.1245
# Semisaturacion de la respuesta inmune (celulas)       
g = 2.019e7
# Tasa de inactivacion de efectoras por tumor (dia-1*celulas-1)       
m = 3.422e-10
# Tasa de lisis tumoral por efectoras (dia-1*celulas-1)     
n = 1.101e-7
# Muerte natural - migracion de efectoras (dia-1)      
d = 0.0412

def sistema(t, y):
    E, T = y
    dEdt = s + (p*E*T)/(g+T) - m*E*T - d*E
    dTdt = a*T*(1-b*T) - n*E*T
    return np.array([dEdt, dTdt])

def euler(f, h, xi, yi, xFin):
    x = xi
    y = np.array(yi)
    xout = [x]
    yout = [y]
    while x < xFin:
        if x + h > xFin:
            h = xFin - x
        y = y + h * f(x, y)
        x = x + h
        xout.append(x)
        yout.append(y)
    return xout, yout

def eulerMejorado(f, h, xi, yi, xFin):
    x = xi
    y = np.array(yi)
    xout = [x]
    yout = [y]
    while x < xFin:
        if x + h > xFin:
            h = xFin - x
        y_pred = y + h * f(x, y)
        y = y + (h/2) * (f(x, y) + f(x + h, y_pred))
        x = x + h
        xout.append(x)
        yout.append(y)
    return xout, yout

def rk4(f, h, xi, yi, xFin):
    x = xi
    y = np.array(yi)
    xout = [x]
    yout = [y]
    while x < xFin:
        if x + h > xFin:
            h = xFin - x
        k1 = f(x,y)
        k2 = f(x + h/2, y + h/2 * k1)
        k3 = f(x + h/2, y + h/2 * k2)
        k4 = f(x + h,y + h * k3)
        y = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        x = x + h
        xout.append(x)
        yout.append(y)
    return xout, yout


# Condiciones iniciales
E0 = 3.3e5
T0 = 1e6
yi = [E0, T0]

# Parametros de simulación
xi  = 0      
xFin = 365   
h = 0.5 

# Correr los tres metodos
tiempoEuler, solucionEuler  = euler(sistema, h, xi, yi, xFin)
tiempoEulerMejorado, solucionEulerMejorado   = eulerMejorado(sistema, h, xi, yi, xFin)
tiempoRK4, solucionRK4 = rk4(sistema, h, xi, yi, xFin)

print("Tiempo final:", tiempoRK4[-1])

print("E final Euler:", solucionEuler[-1][0])
print("T final Euler:", solucionEuler[-1][1])

print("E final Euler Mejorado:", solucionEulerMejorado[-1][0])
print("T final Euler Mejorado:", solucionEulerMejorado[-1][1])

print("E final RK4:", solucionRK4[-1][0])
print("T final RK4:", solucionRK4[-1][1])



# Validacion con scipy
solucionReferencia = solve_ivp(sistema, [xi, xFin], yi, method='RK45', max_step=0.5)

print("E final RK45:", solucionReferencia.y[0][-1])
print("T final RK45:", solucionReferencia.y[1][-1])

# Diferencia entre RK4 casero y RK45 de scipy
print("Diferencia E final RK4 vs RK45:",(solucionRK4[-1][0] - solucionReferencia.y[0][-1]))
print("Diferencia T final RK4 vs RK45:",(solucionRK4[-1][1] - solucionReferencia.y[1][-1]))