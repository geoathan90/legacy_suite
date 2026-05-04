main.py has the main function definitions.

eval.py is used simply to read the charts and interpolate SAG values for various SPANS.

INITIAL ALGORITHM

know sag1, span, temperature1
calculate Th1
calculate Th2 for span, temperature2
compute (xr-Span) for μονόπλευρο

# conductor_trials
Calculate new values for Tension/Sag for different temperatures

Create a GUI for user friendliness.

------------- HOW TO RUN THE GUI ---------------
Step1) go to codespaces
Step2) pip install streamlit (if not already installed)
Step3) streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Step4) go to PORTS tab (next to the TERMINAL on the bottom panel)
Step5) click the globe icon, open in browser