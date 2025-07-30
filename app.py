import streamlit as st
import pandas as pd
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler

st.set_page_config(page_title="QUBO Optimizer", layout="wide")

st.title("📊 QUBO Optimizer")
st.write("Upload a CSV dataset and describe your optimization goal.")

uploaded_file = st.file_uploader("Upload your dataset", type="csv")
objective = st.text_area("What are you trying to optimize?", height=100)

if uploaded_file and objective:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    if st.button("🧠 Simulate Optimization (QUBO → SA)"):

        st.write("Running QUBO simulation with Simulated Annealing...")

        # ⚠️ Toy QUBO logic: use 3 variables with made-up weights and interactions
        Q = {('x1', 'x1'): -1, ('x2', 'x2'): -1, ('x3', 'x3'): -1,
             ('x1', 'x2'): 0.5, ('x2', 'x3'): 0.5, ('x1', 'x3'): 0.5}

        bqm = BinaryQuadraticModel.from_qubo(Q)
        sampler = SimulatedAnnealingSampler()
        result = sampler.sample(bqm, num_reads=100)

        st.subheader("Result (lowest energy sample):")
        best = result.first.sample
        st.json(best)
        st.text("Energy: " + str(result.first.energy))
