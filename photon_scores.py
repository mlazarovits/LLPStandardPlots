import ROOT
import os
from src.plotter import Plotter1D, Plotter2D
from src.style import StyleManager
import cmsstyle as CMS

def format_hist(hist, color, style = 1, normalize = False):
	hist.SetLineColor(color)
	hist.SetLineStyle(style)
	hist.SetLineWidth(2)
	hist.SetFillColor(color)
	hist.SetFillStyle(3003)
	hist.SetStats(0)
	if normalize and hist.Integral() > 0:
		hist.Scale(1.0 / hist.Integral())


#for plotting per-object (photon) observables
def main():
	#initializing style
	lumi = 60
	style = StyleManager(luminosity=lumi)
	style.set_style()
	plotter1d = Plotter1D(style)
	plotter2d = Plotter2D(style)
	sample_label = "MET 2018"
	sample_label_x_pos = 0.69
	plot_format = ".png"	
	
	#processing data
	eosdir = "root://cmseos.fnal.gov//store/user/lpcsusylep/malazaro/KUCMSSkims/skims_v45/"
	files_2018 = ["MET_R18_SVIPM100_v31_MET_AOD_Run2018B_rjrskim_v45.root","MET_R18_SVIPM100_v31_MET_AOD_Run2018A_rjrskim_v45.root","MET_R18_SVIPM100_v31_MET_AOD_Run2018C_rjrskim_v45.root","MET_R18_SVIPM100_v31_MET_AOD_Run2018D_rjrskim_v45.root"]
	files = files_2018
	files = [eosdir+file for file in files]
	chain = ROOT.TChain("kuSkimTree")
	#tchain files
	for file in files:
		chain.Add(file)
	df = ROOT.RDataFrame(chain) 
	
	#plotting overlay of photon beam halo discriminant scores
	df_newbrs = df.Define("bhScore_trueBH","selPho_beamHaloCNNScore[selPho_beamHaloCR == 1]").Define("bhScore_trueGJetsCR","selPho_beamHaloCNNScore[selPho_GJetsCR == 1]").Define("selPhoEta_predBH","selPhoEta[selPho_beamHaloCNNScore > 0.917252]").Define("selPhoWTime_predBH","selPhoWTime[selPho_beamHaloCNNScore > 0.917252]").Define("selPhoEta_predPB","selPhoEta[selPho_physBkgCNNScore > 0.81476355]").Define("selPhoWTime_predPB","selPhoWTime[selPho_physBkgCNNScore > 0.81476355]")

	#create 1d histograms	
	h1 = df_newbrs.Histo1D(("score_trueBH","score_trueBH",50,0,1),"bhScore_trueBH")
	color = style.get_color(0)
	format_hist(h1, color, normalize = True)	
	h2 = df_newbrs.Histo1D(("score_trueGJets","score_trueGJets",50,0,1),"bhScore_trueGJetsCR")
	color = style.get_color(1)
	format_hist(h2, color, normalize = True)

	#create 2d histograms
	h_bh_2d = df_newbrs.Histo2D(("etatime_predBH","etatime_predBH;time;eta",50,-20,20,50,-1.5,1.5),"selPhoWTime_predBH","selPhoEta_predBH")
	h_pb_2d = df_newbrs.Histo2D(("etatime_predPB","etatime_predPB;time;eta",50,-20,-2,50,-1.5,1.5),"selPhoWTime_predPB","selPhoEta_predPB")

	#df.Report()
	
	#make 2D predicted eta-time plots
	name = "eta_time_predBH"
	time_min = -20
	time_max = 20
	eta_min = -1.5
	eta_max = 1.5
	time_label = "Photon time [ns]"
	eta_label = "Pseudorapidity (#eta)"
	pred_bh_canvas = CMS.cmsCanvas(name, time_min, time_max, eta_min, eta_max, time_label, eta_label, 
                              square=False, extraSpace=0.01, iPos=0, with_z_axis=True)
	axis_labels = {}
	axis_labels['x'] = time_label
	axis_labels['y'] = eta_label
	final_state_label = ""
	can, hist = plotter2d.plot_2d_baseFormat(h_bh_2d, pred_bh_canvas, axis_labels, sample_label, final_state_label, sample_label_x_pos=sample_label_x_pos)
	can.cd()
	hist.Draw("colz")
	style.draw_cms_labels(prelim_str="Preliminary")#cms_x=0.16, cms_y=0.93, prelim_str="Preliminary", prelim_x=0.235, lumi_x=0.75, cms_text_size_mult=1.25)
	style.draw_process_label(sample_label, x_pos=sample_label_x_pos, y_pos=0.88)
	can.SaveAs('etatime_predBH'+plot_format)
	
	#predicted pb
	sample_label_x_pos = 0.15
	name = "eta_time_predPB"
	time_max = -2
	pred_pb_canvas = CMS.cmsCanvas(name, time_min, time_max, eta_min, eta_max, time_label, eta_label, 
                              square=False, extraSpace=0.01, iPos=0, with_z_axis=True)
	can2, hist2 = plotter2d.plot_2d_baseFormat(h_pb_2d, pred_pb_canvas, axis_labels, sample_label, final_state_label, sample_label_x_pos=sample_label_x_pos)
	can2.cd()
	hist2.Draw("colz")
	style.draw_cms_labels(prelim_str="Preliminary")#cms_x=0.16, cms_y=0.93, prelim_str="Preliminary", prelim_x=0.235, lumi_x=0.75, cms_text_size_mult=1.25)
	style.draw_process_label(sample_label, x_pos=sample_label_x_pos, y_pos=0.88)
	can2.SaveAs('etatime_predPB'+plot_format)


	sample_label_x_pos = 0.65
	#make 1D score plots
	#setup canvas
	name = "score_can"
	var_label = "selPho_beamHaloCNNScore"
	x_min = 0
	x_max = 1
	can = plotter1d._initialize_canvas(name, x_min, x_max, var_label)
	can.SetLogy()

	#create legend
	legend = CMS.cmsLeg(0.55, 0.7, 0.94, 0.88, textSize=0.035)
	#legend = CMS.cmsLeg(0.35, 0.675, 0.65, 0.874, textSize=0.035)
	legend.AddEntry(h1.GetPtr(), "True Beam Halo", "fl")
	legend.AddEntry(h2.GetPtr(), "True GJets CR", "fl")

	#create cut lines
	line1 = ROOT.TLine(0.917252, 0, 0.917252, 1) #beam halo
	color = style.get_color(0)
	line1.SetLineColor(color)
	line1.SetLineWidth(2)
	line2 = ROOT.TLine((1 - 0.81476355), 0, (1 - 0.81476355), 1) #phys bkg
	color = style.get_color(1)
	line2.SetLineColor(color)
	line2.SetLineWidth(2)
		
	#setup hist axes
	x_label = "Beam Halo Discriminant Score"
	plotter1d.setup_axes(h1, x_label, normalized=True) 
	h1.Draw("HIST")
	h2.Draw("HIST SAME")
	line1.Draw("SAME")
	line2.Draw("SAME")
	#add histograms after dereferencing
	legend.Draw("SAME")
	style.draw_cms_labels(prelim_str="Preliminary")#cms_x=0.16, cms_y=0.93, prelim_str="Preliminary", prelim_x=0.235, lumi_x=0.75, cms_text_size_mult=1.25)
	style.draw_process_label(sample_label, x_pos=sample_label_x_pos, y_pos=0.88)
	can.SaveAs("predscores_MET2018"+plot_format)
	
	
	#plotting eta-time of beamhalo tagged photons






if __name__ == "__main__":
	#import kerebos credentials to conda env if not already there
	kerb = os.getenv("KRB5CCNAME")
	if(kerb is None):
	    print("Setting kerebos credentials")
	    os.environ["KRB5CCNAME"] = "API:"
	main()
