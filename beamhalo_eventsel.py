import os
import ROOT
from src.loader import DataLoader
from src.style import StyleManager
from src.plotter import Plotter1D
from src.plotter import Plotter2D_v2
from pathlib import Path


#import kerebos credentials to conda env if not already there
kerb = os.getenv("KRB5CCNAME")
if(kerb is None):
    print("Setting kerebos credentials")
    os.environ["KRB5CCNAME"] = "API:"
lumi = 138 #Run 2 only
# 1. Setup Style
style = StyleManager(luminosity=lumi)
style.set_style()

# 2. Initialize Loader and Plotter
loader = DataLoader("kuSkimTree", luminosity=lumi)
#plotter = Plotter1D(style)
plotter2d = Plotter2D_v2(style)
plotter1d = Plotter1D(style)

# 3. Load Data
# Unified loader handles file I/O and selection application
#signal_files = ["path/to/signal.root"]
#bg_files = ["path/to/background.root"]
data_files = ["root://cmseos.fnal.gov//store/user/lpcsusylep/malazaro/KUCMSSkims/skims_v45/MET_R18_SVIPM100_v31_MET_AOD_Run2018B_rjrskim_v45.root"]
flags = ["passNPhoGe1SelectionBeamHaloCR"]
data_data, _ = loader.load_data_unified(data_files, flags, [])
# 4. Generate Plot
# Data structure is: data[flag][filename] = {variable: numpy_array}
#current_sig = sig_data["passNHad1SelectionSRTight"]
#for studying beam halo filter
current_data_data = data_data[flags[0]]

sample_label_x_pos = 0.32 # Original logic for signal
sample_label_x_pos = 0.59 # Original logic for combined background
sig_name, fs_label_latex = "", ""
for fname, data in current_data_data.items():
    #sig_name = parse_signal_name(fname)
    sample_label_x_pos = 0.32 # Original logic for signal
    canvas, _ = plotter2d.plot_2d(data, "selPhoWTime", "selPhoEta", f"run2data_2d_{Path(fname).stem}_{flags[0]}", sig_name, fs_label_latex, sample_label_x_pos=sample_label_x_pos)
    #function def in main.py
    #save_canvas(canvas, output_format, f_out, output_dir, plots_2d_subdir)
    
    # 5. Save
    canvas.SaveAs("test.pdf")

