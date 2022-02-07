from scipy.integrate import quad
import numpy as np
from numpy import cos, sin, pi, array, reshape, zeros, dot, linspace
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def integrate(func, a, b):
    """Use QUADPACK to integrate a vector-valued function."""
    r0 = func(a)
    n = len(r0)
    try:
        s = r0.shape
    except AttributeError:
        s = (n,)
    
    integrals = []
    for i in range(n):
        integral = quad(lambda t: func(t)[i], a, b)[0]
        integrals.append(integral)
    
    return reshape(integrals, s)

def ship_pts(x, y, alpha=0):
    """Give verices for asteroid-obliterating ship at pos with angle alpha
    clockwise off vertical.
    
    returns (2,n)-array. Columns are vertices"""
    
    out = np.column_stack([[sin(alpha + 9*pi/8), cos(alpha + 9*pi/8)],
                           [sin(alpha), cos(alpha)],
                           [sin(alpha + 7*pi/8), cos(alpha + 7*pi/8)],
                           [.8*sin(alpha + 7*pi/8), .8*cos(alpha + 7*pi/8)],
                           [.8*sin(alpha + 9*pi/8), .8*cos(alpha + 9*pi/8)],
                           [sin(alpha + 9*pi/8), cos(alpha + 9*pi/8)],
                         ])
    out += np.column_stack([(x,y)]*6)
    return out
    
R = np.array(((0, 1), (-1, 0))) # Rotation cw 90º
    
stalag = np.column_stack([(10, -2), (15, 12), (25, 5), (35, 10), (38, -2)])
stalac = np.column_stack([(5, 18), (9, 10), (15, 17), (28, 12), (35, 14), (38, 18)])
endzone = np.column_stack([(35, 10), (38, -2),  (38, 18), (35, 14)])

def detect(pos):
    """Detect if pos is winning or out-of-bounds."""
    if pos[0] >= 35:
        return 1

    p = pos[:2]
    for i in [0, 2]:
        u,v,w = np.transpose(stalag)[i:i+3]
        n = dot(R,(v - u))
        m = dot(R,(w - v))
        if dot(p, n) > dot(u, n) and dot(p, m) > dot(v, m):
            return -1
    for i in [0, 2]:
        u,v,w = np.transpose(stalac)[i:i+3]
        n = dot(R,(v - u))
        m = dot(R,(w - v))
        if dot(p, n) < dot(u, n) and dot(p, m) < dot(v, m):
            return -1    
    else: 
        return 0
    
def make_run(DURATION, c, evolve):
    state = np.zeros(5)
    clock = 0
    FINISHED = False

    fig, ax = plt.subplots(figsize=(10,5))

    trail = plt.plot([], [], ls= '--', zorder=-2)[0]

    plt.fill(*stalag, color="gray", alpha=.6)
    plt.fill(*stalac, color="gray", alpha=.6)
    plt.fill(*endzone, color="green", alpha=.2)

    plt.grid(True)
    plt.title("")

    tt=plt.text(.5, 15.5, f"$t = {0:.1f}$", fontsize=16)

    ship = plt.fill(*ship_pts(0,0,0), color='b')[0]
    plt.xlim(-2, 38)
    plt.ylim(-2,18)
    plt.gca().set_aspect('equal')
    
    def init_func():
        global clock, state, FINISHED

        state = np.zeros(5)
        clock = 0
        FINISHED = False

        ax.set_title("")
        trail.set_data([],[])

    def update(t):
        """Update the state of the ship at time t."""
        global clock, state, FINISHED

        tt.set_text(f"$t = {t:.1f}$")

        if FINISHED:
            return

        dt = t - clock


        if (win := detect(state[:2])) != 0:
            FINISHED = True
            if win > 0:
                ax.set_title(f"You won in {clock:.3f} seconds!")
            elif win < 0:
                ax.set_title("You met the wall!")
            return

        for j in range(100):
            state = evolve(dt/100, state, c(clock + j*dt/100))


        if c(clock)[0] > 1/2:
            ship.set_color('r')
        else:
            ship.set_color('b')

        # DEBUG: Uncomment to check state at second intervals
        # if np.floor(t) > np.floor(clock):
        #     print(t, state)

        # Update position, trail
        ship.set_xy(np.transpose(ship_pts(*state[:2], alpha=state[4])))

        xs, ys = trail.get_data()    
        xs = np.concatenate([xs,[state[0]]])
        ys = np.concatenate([ys,[state[1]]])

        trail.set_data(xs, ys)

        print('.',end='')
        clock += dt

    anim = FuncAnimation(fig, update, frames=linspace(0, DURATION, DURATION*24 + 1), interval=1000/24, init_func=init_func)

    return anim

__all__ = ['integrate', 'ship_pts', "stalag", "stalac", "endzone", "detect", "make_run"]