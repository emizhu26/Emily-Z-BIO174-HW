# import web app modules
import streamlit as st
import plotly.graph_objs as go
from plotly.graph_objs import Scatter, Layout, Figure, Histogram
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots


def calcNextGen(freqA, WAA, WAa, Waa, pop):
    freqa = 1 - freqA
    fAA = freqA ** 2
    fAa = 2 * freqA * freqa
    faa = freqa ** 2
 
    M = WAA*fAA + WAa*fAa + Waa*faa

    fAA1 = (WAA*fAA) / M
    fAa1 = (WAa*fAa) / M
    faa1 = (Waa*faa) / M

    nAA1 = np.random.binomial(pop, fAA1)
    nAa1 = np.random.binomial(pop, fAa1)
    naa1 = np.random.binomial(pop, faa1)

    fA1 = (nAA1 + (1/2) * nAa1) / (nAA1+nAa1+naa1)

    return fA1


def simulate(initA, pop, WAA, WAa, Waa, gen, sim):
    plots = make_subplots(rows=2, cols=1)
    fig = go.Figure()
    a = np.zeros(shape=(sim, gen+1))
    b = np.array(range(gen+1))

    hist = []
    for x in range(sim):
        A_vals = []
        A_vals.append(initA)
        beginA = initA
        for i in range(gen):
            nextA = calcNextGen(beginA, WAA, WAa, Waa, pop)
            A_vals.append(nextA)
            beginA = nextA
        plots.add_trace(Scatter(x = b, y = A_vals, mode = 'lines',))
        a[x] = A_vals
        hist.append(A_vals[-1])

        A_vals = []
        A_vals.append(initA)

    fig2 = Histogram(x=hist, opacity=0.5, autobinx=False, xbins=dict(start=-0.01, end=1.01, size=0.01))
    plots.append_trace(fig2, 2, 1)
    plots['layout']['xaxis2'].update(range=[-0.01, 1.01])
    plots['layout']['yaxis1'].update(range=[0, 1])
    plots['layout'].update(height=600, width=800)
    plots['layout'].update(showlegend=False)
    return plots


_, right_column = st.columns([1,2])


nSim = st.sidebar.slider('Number of Simulations:',min_value=1,max_value=100,step=1)
AA = st.sidebar.slider('Fitness of AA:',min_value=0.,max_value=1.,step=0.05,value=1.)
Aa = st.sidebar.slider('Fitness of Aa:',min_value=0.,max_value=1.,step=0.05,value=1.)
aa = st.sidebar.slider('Fitness of aa:',min_value=0.,max_value=1.,step=0.05,value=1.)
pop = st.sidebar.select_slider('Population Size:',[10,50,100,500,1000])
gen = st.sidebar.slider("Number of Generations:",min_value=100,max_value=1000,step=100)
stA = st.sidebar.slider("Starting frequency of A:", min_value=0.01,max_value=0.99,step=0.01, value=0.5)

with right_column:
    plot = simulate(stA, pop, AA, Aa, aa, gen, nSim)
    st.write(plot)