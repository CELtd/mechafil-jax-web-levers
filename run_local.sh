#!/bin/bash

source activate cel
streamlit run mechafil_jax_web_levers/Filecoin_CryptoEconomics.py --server.runOnSave True --server.allowRunOnSave True --server.headless True
