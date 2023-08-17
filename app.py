import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Petient Survival Prediction App",
                   page_icon="🛏", layout="wide")

model=load_model('models/final_model.h5')


st.markdown("<h1 style='text-align: center;'>Higg's Boson Event Detection</h1>", unsafe_allow_html=True)
def main():
    with st.form('predictin_form'):
        st.subheader("Enter the below Features:")

        DER_deltar_tau_lep=st.number_input("DER_deltar_tau_lep Value: ",0.75,4.30,format="%.2f")
        DER_mass_vis=st.number_input("DER_mass_vis Value: ",17.00,132.00,format="%.2f")
        PRI_tau_pt=st.number_input("PRI_tau_pt Value: ",20.00,64.00,format="%.2f")
        DER_sum_pt=st.number_input("DER_sum_pt Value: ",46.00,133.00,format="%.2f")
        DER_pt_ratio_lep_tau=st.number_input("DER_pt_ratio_lep_tau Value: ",0.40,3.00,format="%.2f")
        PRI_met=st.number_input("PRI_met Value: ",0.10,73.00,format="%.2f")
        DER_mass_transverse_met_lep=st.number_input("DER_mass_transverse_met_lep Value: ",0.00,146.00,format="%.2f")
        DER_mass_MMC=st.number_input("DER_mass_MMC Value: ",43.00,183.00,format="%.2f")
        
        
        submit=st.form_submit_button("Predict")
    if submit:
        DER_deltar_tau_lep=DER_deltar_tau_lep
        DER_mass_vis=DER_mass_vis
        PRI_tau_pt=PRI_tau_pt
        DER_sum_pt=DER_sum_pt
        DER_pt_ratio_lep_tau=DER_pt_ratio_lep_tau
        PRI_met=PRI_met
        DER_mass_transverse_met_lep=DER_mass_transverse_met_lep
        DER_mass_MMC=DER_mass_MMC

        value=[DER_deltar_tau_lep,DER_mass_vis,PRI_tau_pt,DER_sum_pt,
                       DER_pt_ratio_lep_tau,PRI_met,DER_mass_transverse_met_lep,DER_mass_MMC
                       ]

        # final_values=value_1+value_3
        # final_values=np.array([value]).reshape(1,8)

        pred=model.predict(np.array([value]))[0][0]

        if pred<=5:
            d='Background'
        else:
            d='Signal'

        st.write(f"This is a {d} event.")

if __name__=='__main__':
    main()