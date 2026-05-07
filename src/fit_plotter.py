import numpy as np
import uproot
import yaml
import ROOT
from collections import namedtuple
from src.style import StyleManager

# ── Bin label tables ──────────────────────────────────────────────────────────

RISR_LABELS = {
    #"00": "[0.7, 0.8)",
    "00": "< 0.3",
    "10": "< 0.3",
    "01": "#geq 0.3",
    "11": "#geq 0.3",
    "22": "#geq 0.4",
    "12": "#geq 0.4",
    "02": "#geq 0.4",
    "21": "[0.3, 0.35)",
    "20": "#geq 0.9",
}

MSRS_MS_LABELS = {
    "0": "M_{S}^{SR,-}",
    "1": "M_{S}^{SR,+}",
    "2": "M_{S}^{SR,++}",
}
MSRS_RS_LABELS = {
    "0": "R_{S}^{-}",
    "1": "R_{S}^{+}",
    "2": "R_{S}^{++}",
}

MS_DELAYED_LABELS = {
    "00": "#geq 2000",
}

COMBINED_MS_DELAYED_LABELS = {
    "00": "#geq 2000",
    #"10": "M_{S}^{SR}",
}

CHANNEL_LABELS = {
    #Full NonCompressed Regions
    "Ch1CRGeLep1": "#geq 1 Lep. SV, S^{-}_{xy}",
    "Ch2SRGeLep1": "#geq 1 Lep. SV, S^{+}_{xy}",
    "Ch3CRGeHad1": "#geq 1 Had. SV, S^{-}_{xy}",
    "Ch4SRGeHad1": "#geq 1 Had. SV, S^{+}_{xy}",
    "Ch5CRgeq1PhoBHEarlyBin":"BH #gamma, early timing",
    "Ch6CRgeq1PhoBHLateBin": "BH #gamma, late timing",
    "Ch7CRgeq1PhoNotBHEarlyBin": "Non-BH #gamma, early timing",
    "Ch8SRgeq1PhoNotBHLateBin":"Non-BH #gamma, late timing",
    "Ch8CRgeq1PhoNotBHLateBin":"Non-BH #gamma, late timing (VR)",
    "Ch9CReq1PhoMedIsoPromptBin":"Med Iso 1 #gamma",
    "Ch10SReq1PhoTightIsoPromptBin":"Tight Iso 1 #gamma",
    "Ch11CReq2PhoMedIsoPromptBin":"Med Iso 2 #gamma",
    "Ch12SReq2PhoTightIsoPromptBin":"Tight Iso 2 #gamma",
    # Compressed shape_transfer
    "Ch1CRHadLow":        "1 Had. SV, d_{xy}/#sigma_{d_{xy}} #in [100, 600)",
    "Ch2CRHadHigh":       "1 Had. SV, d_{xy}/#sigma_{d_{xy}} #in [600, 1000)",
    "Ch3CRPhoIso1":       "Prompt #gamma, ISO #in [0.4, 0.5)",
    "Ch4CRPhoIso2":       "Prompt #gamma, ISO #in [0.5, 0.52)",
    # Uncompressed shape_transfer
    "Ch5CRPho1Iso":       "Prompt #gamma, ISO #in [0.4, 0.5)",
    "Ch6CRPho2Iso":       "Prompt #gamma, ISO #in [0.5, 0.52)",
    "Ch7CRHadLow":        "1 Had. SV, d_{xy}/#sigma_{d_{xy}} #in [100, 600)",
    "Ch8CRHadHigh":       "1 Had. SV, d_{xy}/#sigma_{d_{xy}} #in [600, 1000)",
    # Uncompressed ABCD
    "Ch1CRgeq1PhoBHEarly":    "BH #gamma, early timing",
    "Ch2CRgeq1PhoBHLate":     "BH #gamma, late timing",
    "Ch3CRgeq1PhoNotBHEarly": "Non-BH #gamma, early timing",
    "Ch4CRgeq1PhoNotBHLate":  "Non-BH #gamma, late timing",
    "Ch1CRgeq1PhoBHEarlyDxyHadLow":    "Early BH #gamma, #geq 1 SV S^{-}_{xy}",
    "Ch2CRgeq1PhoBHLateDxyHadLow":     "Late BH #gamma, #geq 1 SV S^{-}_{xy}",
    "Ch3CRgeq1PhonotBHEarlyDxyHadLow": "Early Non-BH #gamma, #geq 1 SV S^{-}_{xy}",
    "Ch4CRgeq1PhonotBHLateDxyHadHigh":  "Late Non-BH #gamma, #geq 1 SV S^{+}_{xy}",
    "Ch5CRgeq1PhoMedIsoPromptBin":  "Med Iso #geq 1 #gamma",
    "Ch5CReq1PhoMedIsoPromptBin":  "Med Iso 1 #gamma",
    "Ch5CReq2PhoMedIsoPromptBin":  "Med Iso 2 #gamma",
    "Ch6SReq1PhoTightIsoPromptBin": "Tight Iso 1 #gamma",
    "Ch6SReq1PhoVeryTightIsoPromptBin": "Very Tight Iso 1 #gamma",
    "Ch6SReq2PhoTightIsoPromptBin": "Tight Iso 2 #gamma",
    "Ch5CReq1PhoMedIsoPromptSVLowDxy":"1 Prompt Med Iso #gamma, #geq 1 SV S^{-}_{xy}",
    "Ch6SReq1PhoTightIsoPromptSVHighDxy":"1 Prompt Tight Iso #gamma, #geq 1 SV S^{+}_{xy}",
    "Ch5CReq2PhoMedIsoPromptSVLowDxy":"2 Prompt Med Iso #gamma, #geq 1 SV S^{-}_{xy}",
    "Ch6SReq2PhoTightIsoPromptSVHighDxy":"2 Prompt Tight Iso #gamma, #geq 1 SV S^{+}_{xy}",
    "Ch5CRgeq1PhoMedIsoPromptSVLowDxy":"#geq 1 Prompt Med Iso #gamma, #geq 1 SV S^{-}_{xy}",
    "Ch6SRgeq1PhoTightIsoPromptSVHighDxy":"#geq 1 Tight Iso #gamma, #geq 1 SV S^{+}_{xy}"		
}

# Compact labels used only in the combined plot where space is tight.
# ABCD channels use the comma-split two-line feature: "group, sub" → stacked.
COMBINED_CHANNEL_LABELS = {
    # ABCD
    "Ch1CRPhoBHEarly":    "#gamma^{ BH}, t_{ -}",
    "Ch2CRPhoBHLate":     "#gamma^{ BH}, t_{ +}",
    "Ch3CRPhonotBHEarly": "#gamma^{ !BH}, t_{ -}",
    "Ch4CRPhonotBHLate":  "#gamma^{ !BH}, t_{ +}",
    # Shape transfer
    "Ch5CRPho1Iso":       "#gamma^{ iso,1}",
    "Ch6CRPho2Iso":       "#gamma^{ iso,2}",
    "Ch7CRHadLow":        "SV_{qq}^{CR,-}",
    "Ch8CRHadHigh":       "SV_{qq}^{CR,+}",
}

# ── Parsed fit configuration ──────────────────────────────────────────────────

FitConfig = namedtuple("FitConfig", [
    "mode",            # 'compressed' | 'uncompressed'
    "shape_bin_order", # [(ch, [bins]), ...]
    "shape_pairs",     # {anchor_ch: [buoy_chs]}
    "anchor_bin",      # '00' or '01'
    "abcd_bin_order",  # [(ch, [bins]), ...] or []
    "abcd_pairs",      # {sr_ch: [cr_chs]} or {}
])

_canvas_counter = [0]  # mutable counter for unique ROOT names


def _uid():
    _canvas_counter[0] += 1
    return _canvas_counter[0]


# ── Main class ────────────────────────────────────────────────────────────────

class FitPlotter:
    """
    Produces pre/postfit comparison plots from a Combine FitDiagnostics output.

    Handles compressed (1D RISR) and uncompressed (2D Ms×Rs + ABCD) fits.

    Output per call to plot_all():
      Compressed:
        {prefix}_shape_{pre,post}fit.{fmt}
        {prefix}_pair_{A}_{B}_together_{pre,post}fit.{fmt}
        {prefix}_pair_{A}_{B}_sidebyside_{pre,post}fit.{fmt}
      Uncompressed (adds):
        {prefix}_abcd_flat_{pre,post}fit.{fmt}
        {prefix}_abcd_grid_{pre,post}fit.{fmt}
        {prefix}_combined_{pre,post}fit.{fmt}
    """

    def __init__(self, luminosity=136, energy=13):
        self.style = StyleManager(luminosity=luminosity, energy=energy)
        self.style.set_style()
        self.luminosity = luminosity
        self.energy = energy

    # ── Public API ────────────────────────────────────────────────────────────

    def plot_all(self, fit_result_path, fit_config_path, output_prefix="fit",
                 output_format="pdf", mode_override=None, show_sr=False):
        cfg = self._load_config(fit_config_path, mode_override)
        f   = uproot.open(fit_result_path)

        print(f"[FitPlotter] mode={cfg.mode} | "
              f"{len(cfg.shape_bin_order)} shape channels"
              + (f" | {len(cfg.abcd_bin_order)} ABCD channels" if cfg.abcd_bin_order else ""))

        bin_scheme     = "msrs" if cfg.mode == "uncompressed" else "risr"
        shape_all_bins = [b for _, bins in cfg.shape_bin_order for b in bins]
        # Collect (canvas_pre, canvas_post, short_name) for all plots, then
        # flush to a single ROOT file or individual image files at the end.
        plots = []
        pre_shape = None
        post_shape = None
        if len(shape_all_bins) > 0:
            pre_shape  = self._extract_yields(f, "shapes_prefit", shape_all_bins)
            post_shape = self._extract_yields(f, "shapes_fit_b",  shape_all_bins)



            # ── comprehensive shape_transfer ──────────────────────────────────────
            shape_deco = self._build_decorations(cfg.shape_bin_order, bin_scheme)
            plots.append((
                self._draw_standard_canvas(*pre_shape,  shape_deco, f"shp_pre_{_uid()}",  "Prefit"),
                self._draw_standard_canvas(*post_shape, shape_deco, f"shp_pst_{_uid()}", "Postfit"),
                "shape",
            ))

            # ── per-pair plots ────────────────────────────────────────────────────
            for anchor_ch, buoy_chs in cfg.shape_pairs.items():
                anchor_bins = dict(cfg.shape_bin_order)[anchor_ch]
                for buoy_ch in buoy_chs:
                    buoy_bins = dict(cfg.shape_bin_order)[buoy_ch]
                    ai = shape_all_bins.index(anchor_bins[0])
                    bi = shape_all_bins.index(buoy_bins[0])
                    na, nb = len(anchor_bins), len(buoy_bins)

                    a_pre  = tuple(arr[ai:ai+na] for arr in pre_shape)
                    a_post = tuple(arr[ai:ai+na] for arr in post_shape)
                    b_pre  = tuple(arr[bi:bi+nb] for arr in pre_shape)
                    b_post = tuple(arr[bi:bi+nb] for arr in post_shape)

                    tag       = f"{self._ch_short(anchor_ch)}_{self._ch_short(buoy_ch)}"
                    pair_ord  = [(anchor_ch, anchor_bins), (buoy_ch, buoy_bins)]
                    pair_deco = self._build_decorations(pair_ord, bin_scheme)
                    tog_pre   = tuple(np.concatenate([a, b]) for a, b in zip(a_pre,  b_pre))
                    tog_post  = tuple(np.concatenate([a, b]) for a, b in zip(a_post, b_post))

                    plots.append((
                        self._draw_standard_canvas(*tog_pre,  pair_deco, f"tog_pre_{_uid()}",  "Prefit"),
                        self._draw_standard_canvas(*tog_post, pair_deco, f"tog_pst_{_uid()}", "Postfit"),
                        f"pair_{tag}_together",
                    ))

                    a_deco = self._build_decorations([(anchor_ch, anchor_bins)], bin_scheme)
                    b_deco = self._build_decorations([(buoy_ch,   buoy_bins)],   bin_scheme)
                    plots.append((
                        self._draw_sidebyside_canvas(*a_pre,  *b_pre,  a_deco, b_deco,
                                                     f"sbs_pre_{_uid()}",  "Prefit"),
                        self._draw_sidebyside_canvas(*a_post, *b_post, a_deco, b_deco,
                                                     f"sbs_pst_{_uid()}", "Postfit"),
                        f"pair_{tag}_sidebyside",
                    ))

        # ── ABCD plots (uncompressed only) ────────────────────────────────────
        if cfg.abcd_bin_order:
            abcd_all_bins = [b for _, bins in cfg.abcd_bin_order for b in bins]
            pre_abcd  = self._extract_yields(f, "shapes_prefit", abcd_all_bins)
            post_abcd = self._extract_yields(f, "shapes_fit_b",  abcd_all_bins)
            abcd_flat_deco = self._build_decorations(cfg.abcd_bin_order, "ms_delayed")
            plots.append((
                self._draw_standard_canvas(*pre_abcd,  abcd_flat_deco, f"abf_pre_{_uid()}",  "Prefit"),
                self._draw_standard_canvas(*post_abcd, abcd_flat_deco, f"abf_pst_{_uid()}", "Postfit"),
                "abcd_flat",
            ))

            sr_ch = (next(iter(cfg.abcd_pairs)) if cfg.abcd_pairs else None) if show_sr else None
            abcd_grid_deco = self._build_abcd_grid_decorations(cfg.abcd_bin_order, sr_ch=sr_ch)
            plots.append((
                self._draw_standard_canvas(*pre_abcd,  abcd_grid_deco, f"abg_pre_{_uid()}",  "Prefit"),
                self._draw_standard_canvas(*post_abcd, abcd_grid_deco, f"abg_pst_{_uid()}", "Postfit"),
                "abcd_grid",
            ))
            if pre_shape is not None and post_shape is not None:
                comb_deco  = self._build_combined_decorations(cfg.abcd_bin_order, cfg.shape_bin_order, sr_ch=sr_ch)
                comb_pre   = tuple(np.concatenate([a, s]) for a, s in zip(pre_abcd,  pre_shape))
                comb_post  = tuple(np.concatenate([a, s]) for a, s in zip(post_abcd, post_shape))
                plots.append((
                    self._draw_standard_canvas(*comb_pre,  comb_deco, f"cmb_pre_{_uid()}",  "Prefit",  right_panel=True),
                    self._draw_standard_canvas(*comb_post, comb_deco, f"cmb_pst_{_uid()}", "Postfit", right_panel=True),
                    "combined",
                ))

        self._flush(plots, output_prefix, output_format)

    # ── Config loading ────────────────────────────────────────────────────────

    def _load_config(self, path, mode_override=None):
        with open(path) as fh:
            cfg = yaml.safe_load(fh)

        stf            = cfg.get("shape_transfer_fit", {})
        shape_bin_ass  = {} 
        shape_ch_ass   = {} 
        anchor_bin     = None 
        if stf is not None:
            shape_bin_ass  = stf.get("bin_association",    {})
            shape_ch_ass   = stf.get("channel_association", {})
            anchor_bin     = stf.get("anchor_bin", "00")

        abcd           = cfg.get("ABCD_fit") or {}
        abcd_bin_ass   = {} 
        abcd_ch_ass    = {} 
        if abcd is not None:
            abcd_bin_ass   = abcd.get("bin_association",    {})
            abcd_ch_ass    = abcd.get("channel_association", {})

        mode = mode_override or self._detect_mode(cfg)

        return FitConfig(
            mode=mode,
            shape_bin_order=list(shape_bin_ass.items()),
            shape_pairs=shape_ch_ass,
            anchor_bin=anchor_bin,
            abcd_bin_order=list(abcd_bin_ass.items()),
            abcd_pairs=abcd_ch_ass,
        )

    def _detect_mode(self, cfg):
        assoc = cfg.get("shape_transfer_fit", {})
        if assoc is None:
            return "uncompressed"
        assoc = assoc.get("bin_association", {})
        if not assoc:
            return "compressed"
        n = len(next(iter(assoc.values())))
        return "uncompressed" if n == 4 else "compressed"

    # ── Yield extraction ──────────────────────────────────────────────────────

    def _extract_yields(self, uf, folder, bin_list):
        bkg, berr, dy, eyl, eyh = [], [], [], [], []
        for bn in bin_list:
            h = uf[f"{folder}/{bn}/total_background"]
            g = uf[f"{folder}/{bn}/data"]
            bkg.append(float(h.values()[0]))
            berr.append(float(h.errors()[0]))
            dy.append(float(g.member("fY")[0]))
            eyl.append(float(g.member("fEYlow")[0]))
            eyh.append(float(g.member("fEYhigh")[0]))
        return (np.array(bkg), np.array(berr),
                np.array(dy),  np.array(eyl), np.array(eyh))

    # ── Decoration builders ───────────────────────────────────────────────────

    def _build_decorations(self, bin_order, bin_scheme, label_map=None, bin_label_map=None):
        """
        Build decoration dict for a standard canvas.

        bin_scheme: 'risr' | 'msrs' | 'ms_delayed'

        Returns dict with keys:
            n_bins, bin_labels, group_labels, sub_group_labels,
            separator_bins, sub_sep_bins,
            section_labels, section_separator,
            x_axis_title, bottom_margin
        """
        bin_labels       = []
        group_labels     = []
        sub_group_labels = []
        separator_bins   = []
        sub_sep_bins     = []
        cursor           = 0

        for i, (ch, bins) in enumerate(bin_order):
            ch_start = cursor
            prev_ms  = None  # reset per channel for msrs

            if bin_scheme == "msrs":
                for bin_name in bins:
                    suf = bin_name[-2:]
                    ms, rs = suf[0], suf[1]
                    if ms != prev_ms:
                        if prev_ms is not None:
                            sub_group_labels[-1]["end"] = cursor
                            sub_sep_bins.append(cursor)
                        sub_group_labels.append(
                            {"text": MSRS_MS_LABELS.get(ms, ms), "start": cursor, "end": None}
                        )
                        prev_ms = ms
                    bin_labels.append(MSRS_RS_LABELS.get(rs, rs))
                    cursor += 1
                if sub_group_labels and sub_group_labels[-1]["end"] is None:
                    sub_group_labels[-1]["end"] = cursor

            elif bin_scheme == "risr":
                for bin_name in bins:
                    bin_labels.append(RISR_LABELS.get(bin_name[-2:], bin_name[-2:]))
                    cursor += 1

            elif bin_scheme == "ms_delayed":
                _bll = bin_label_map if bin_label_map is not None else MS_DELAYED_LABELS
                for bin_name in bins:
                    bin_labels.append(_bll.get(bin_name[-2:], bin_name[-2:]))
                    cursor += 1

            else:
                for _ in bins:
                    bin_labels.append("")
                    cursor += 1

            _labels = label_map if label_map is not None else CHANNEL_LABELS
            group_labels.append({
                "text":  _labels.get(ch, CHANNEL_LABELS.get(ch, ch)),
                "start": ch_start,
                "end":   cursor,
            })
            if i < len(bin_order) - 1:
                separator_bins.append(cursor)

        has_sub       = bool(sub_group_labels)
        bottom_margin = 0.52 if has_sub else 0.44

        x_titles = {
            "risr":       "R_{ISR}",
            "ms_delayed": "M_{S} [TeV]",
        }
        sg_titles = {}

        return {
            "n_bins":              cursor,
            "bin_labels":          bin_labels,
            "group_labels":        group_labels,
            "sub_group_labels":    sub_group_labels if has_sub else None,
            "sub_group_axis_title": sg_titles.get(bin_scheme, ""),
            "separator_bins":      separator_bins,
            "sub_sep_bins":        sub_sep_bins,
            "section_labels":      None,
            "section_separator":   None,
            "x_axis_title":        x_titles.get(bin_scheme, ""),
            "bottom_margin":       bottom_margin,
            "bin_scheme":          bin_scheme,
        }

    def _build_abcd_grid_decorations(self, abcd_bin_order, sr_ch=None):
        """
        3-level decoration for the ABCD 2×2 grid layout:
          Level 1 (group_labels):     BH γ  |  Non-BH γ
          Level 2 (sub_group_labels): Early | Late  (within each major group)
          Level 3 (bin_labels):       Ms-delayed ranges
        """
        bh_chs    = [(ch, b) for ch, b in abcd_bin_order if "notBH" not in ch and "BH" in ch]
        notbh_chs = [(ch, b) for ch, b in abcd_bin_order if "notBH" in ch]

        bin_labels       = []
        group_labels     = []
        sub_group_labels = []
        separator_bins   = []
        sub_sep_bins     = []
        sr_bins          = []
        cursor           = 0

        for gi, (grp_text, channels) in enumerate(
            [("BH #gamma", bh_chs), ("Non-BH #gamma", notbh_chs)]
        ):
            grp_start = cursor
            for ci, (ch, bins) in enumerate(channels):
                ch_start = cursor
                timing   = "Early" if "Early" in ch else "Late"
                for bin_name in bins:
                    bin_labels.append(MS_DELAYED_LABELS.get(bin_name[-2:], bin_name[-2:]))
                    if ch == sr_ch:
                        sr_bins.append(cursor)
                    cursor += 1
                sub_group_labels.append({"text": timing, "start": ch_start, "end": cursor})
                if ci < len(channels) - 1:
                    sub_sep_bins.append(cursor)

            group_labels.append({"text": grp_text, "start": grp_start, "end": cursor})
            if gi < 1:
                separator_bins.append(cursor)

        return {
            "n_bins":           cursor,
            "bin_labels":       bin_labels,
            "group_labels":     group_labels,
            "sub_group_labels": sub_group_labels,
            "separator_bins":   separator_bins,
            "sub_sep_bins":     sub_sep_bins,
            "section_labels":      None,
            "section_separator":   None,
            "x_axis_title":        "M_{S} [TeV]",
            "sub_group_axis_title": "",
            "bottom_margin":       0.52,
            "sr_bins":             sr_bins,
        }

    def _build_combined_decorations(self, abcd_bin_order, shape_bin_order, sr_ch=None):
        """
        Combined ABCD + shape_transfer decorations with section labels and
        a heavy separator between the two fit components.
        ABCD bins come first, shape_transfer second.
        """
        abcd_deco  = self._build_decorations(abcd_bin_order,  "ms_delayed", COMBINED_CHANNEL_LABELS, COMBINED_MS_DELAYED_LABELS)
        shape_deco = self._build_decorations(shape_bin_order, "msrs",       COMBINED_CHANNEL_LABELS)
        n_abcd     = abcd_deco["n_bins"]
        n_total    = n_abcd + shape_deco["n_bins"]

        def _offset(lst, off):
            return [{"text": g["text"], "start": g["start"] + off, "end": g["end"] + off}
                    for g in lst] if lst else []

        sr_bins = []
        if sr_ch:
            for gl, (ch, _) in zip(abcd_deco["group_labels"], abcd_bin_order):
                if ch == sr_ch:
                    sr_bins.extend(range(gl["start"], gl["end"]))

        return {
            "n_bins":      n_total,
            "bin_labels":  abcd_deco["bin_labels"] + shape_deco["bin_labels"],
            "group_labels": (abcd_deco["group_labels"]
                             + _offset(shape_deco["group_labels"], n_abcd)),
            "sub_group_labels": (_offset(shape_deco["sub_group_labels"] or [], n_abcd) or None),
            "sub_group_axis_title": "",
            "separator_bins": (abcd_deco["separator_bins"]
                               + [s + n_abcd for s in shape_deco["separator_bins"]]),
            "sub_sep_bins": [s + n_abcd for s in shape_deco["sub_sep_bins"]],
            "section_labels":    None,
            "section_separator": n_abcd,
            "x_axis_title":  "",
            "bottom_margin": 0.52,
            "sr_bins":       sr_bins,
        }

    # ── Canvas drawing: standard ──────────────────────────────────────────────

    def _draw_standard_canvas(self, bkg_vals, bkg_errs, data_y, data_eyl, data_eyh,
                               deco, name, title, right_panel=False):
        n      = deco["n_bins"]
        bot_m  = deco.get("bottom_margin", 0.38)
        left_m = 0.10
        right_m = 0.16 if right_panel else 0.04
        split   = 0.30
        has_sub  = bool(deco.get("sub_group_labels"))
        has_sect = bool(deco.get("section_labels"))

        # Scale canvas width with bin count
        cw = max(1200, min(80 * n, 2400))
        canvas = ROOT.TCanvas(f"c_{name}", title, cw, 700)
        canvas.SetFillColor(0)

        pad1 = ROOT.TPad(f"p1_{name}", "main",  0, split, 1, 1)
        pad1.SetBottomMargin(0.02)
        pad1.SetTopMargin(0.17 if has_sect else 0.13)
        pad1.SetLeftMargin(left_m); pad1.SetRightMargin(right_m)
        pad1.SetTicks(1, 1)
        pad1.SetLogy(True); pad1.SetGridx(True); pad1.Draw()

        pad2 = ROOT.TPad(f"p2_{name}", "ratio", 0, 0, 1, split)
        pad2.SetTopMargin(0.02); pad2.SetBottomMargin(bot_m)
        pad2.SetLeftMargin(left_m); pad2.SetRightMargin(right_m)
        pad2.SetTicks(1, 1)
        pad2.SetGridx(True); pad2.SetGridy(True); pad2.Draw()

        # ── histograms ────────────────────────────────────────────────────────
        h_bkg = ROOT.TH1F(f"hb_{name}", "", n, 0, n)
        h_bkg.SetDirectory(0); h_bkg.Sumw2()
        for i in range(n):
            h_bkg.SetBinContent(i + 1, bkg_vals[i])
            h_bkg.SetBinError(i + 1,   bkg_errs[i])
        h_bkg.SetFillColor(ROOT.kAzure - 9)
        h_bkg.SetLineColor(ROOT.kBlue + 1); h_bkg.SetLineWidth(2); h_bkg.SetStats(0)

        h_bkg_err = h_bkg.Clone(f"hberr_{name}")
        h_bkg_err.SetDirectory(0)
        h_bkg_err.SetFillColor(ROOT.kGray + 1)
        h_bkg_err.SetFillStyle(3345)
        h_bkg_err.SetMarkerSize(0)
        h_bkg_err.SetLineColor(0)

        g_data = ROOT.TGraphAsymmErrors(n)
        for i in range(n):
            g_data.SetPoint(i, i + 0.5, data_y[i])
            g_data.SetPointError(i, 0, 0, data_eyl[i], data_eyh[i])
        g_data.SetMarkerStyle(20); g_data.SetMarkerSize(1.2)
        g_data.SetMarkerColor(ROOT.kBlack); g_data.SetLineColor(ROOT.kBlack); g_data.SetLineWidth(2)

        h_ratio = ROOT.TH1F(f"hr_{name}", "", n, 0, n)
        h_ratio.SetDirectory(0)
        h_rband = ROOT.TH1F(f"hrb_{name}", "", n, 0, n)
        h_rband.SetDirectory(0)
        for i in range(n):
            b = bkg_vals[i]
            h_ratio.SetBinContent(i + 1, data_y[i] / b if b > 0 else 0)
            h_ratio.SetBinError(i + 1,   max(data_eyl[i], data_eyh[i]) / b if b > 0 else 0)
            h_rband.SetBinContent(i + 1, 1.0)
            h_rband.SetBinError(i + 1,   bkg_errs[i] / b if b > 0 else 0)
        h_rband.SetFillColor(ROOT.kGray + 1); h_rband.SetFillStyle(3345)
        h_rband.SetMarkerSize(0); h_rband.SetLineColor(0)

        # ── main pad ──────────────────────────────────────────────────────────
        pad1.cd()
        pos_vals = np.concatenate([bkg_vals, data_y])
        pos_vals = pos_vals[pos_vals > 0]
        min_v = max(0.5, 0.3 * pos_vals.min()) if pos_vals.size else 0.5
        max_v = max(float((bkg_vals + bkg_errs).max()),
                    float((data_y   + data_eyh).max()))
        h_bkg.SetMinimum(min_v); h_bkg.SetMaximum(max_v * 10.0)

        h_bkg.GetYaxis().SetTitle("Events / bin")
        h_bkg.GetYaxis().SetTitleSize(0.075); h_bkg.GetYaxis().SetTitleOffset(0.65)
        h_bkg.GetYaxis().SetLabelSize(0.065); h_bkg.GetYaxis().CenterTitle(True)
        h_bkg.GetYaxis().SetTickLength(0.015)
        h_bkg.GetXaxis().SetLabelSize(0);     h_bkg.GetXaxis().SetTickLength(0.015)
        h_bkg.GetXaxis().SetNdivisions(n, 0, 0, False)
        sr_bins = deco.get("sr_bins", [])
        h_sr = None
        if sr_bins:
            h_sr = ROOT.TH1F(f"hsr_{name}", "", n, 0, n)
            h_sr.SetDirectory(0)
            for i in sr_bins:
                h_sr.SetBinContent(i + 1, bkg_vals[i])
                h_sr.SetBinError(i + 1,   bkg_errs[i])
            h_sr.SetFillColor(ROOT.kOrange - 3)
            h_sr.SetLineColor(ROOT.kBlue + 1); h_sr.SetLineWidth(2); h_sr.SetStats(0)

        h_bkg.Draw("HIST")
        if h_sr:
            h_sr.Draw("HIST SAME")
        h_bkg_err.Draw("E2 SAME")
        g_data.Draw("PZ SAME")

        rp = 1.0 - right_m  # left edge of right-margin panel in pad1 NDC
        if right_panel:
            leg = ROOT.TLegend(rp + 0.01, 0.72, 0.995, 0.87)
        else:
            leg = ROOT.TLegend(0.77, 0.42, 1., 0.7)
        leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.055)
        leg.AddEntry(g_data,   "Data",         "lp")
        leg.AddEntry(h_bkg,    f"{title} bkg", "F")
        if h_sr:
            leg.AddEntry(h_sr, "Pred. SR",     "F")
        leg.Draw()

        rp_objs = []
        if right_panel:
            defs = [
                ("M_{S}^{CR,-}", "[0.7, 1) TeV"),
                ("M_{S}^{CR,+}", "#geq 1 TeV"),
                ("M_{S}^{SR,-}", "[1, 1.5) TeV"),
                ("M_{S}^{SR,+}", "#geq 1.5 TeV"),
                ("R_{S}^{-}",    "[0.15, 0.2)"),
                ("R_{S}^{+}",    "#geq 0.2"),
                #("#gamma^{iso,1}", "ISO #in [0.4, 0.5)"),
                #("#gamma^{iso,2}", "ISO #in [0.5, 0.52)"),
                #("SV_{qq}^{CR,-}", "d_{xy}/#sigma_{d} #in [100, 600)"),
                #("SV_{qq}^{CR,+}", "d_{xy}/#sigma_{d} #in [600, 1000)"),
            ]
            lt_key = ROOT.TLatex(); lt_key.SetNDC(True)
            lt_key.SetTextFont(42); lt_key.SetTextSize(0.052); lt_key.SetTextAlign(12)
            y0 = 0.65
            dy = 0.075
            for i, (sym, rng) in enumerate(defs):
                lt_key.DrawLatex(rp + 0.01, y0 - i * dy, f"{sym}  :  {rng}")
            rp_objs.append(lt_key)

        if right_panel:
            self.style.draw_cms_labels(
                cms_x=0.10, cms_y=0.89, prelim_str="Preliminary",
                prelim_x=0.175, lumi_x=0.84, cms_text_size_mult=1.92
            )
        else:
            self.style.draw_cms_labels(
                cms_x=0.10, cms_y=0.89, prelim_str="Preliminary",
                prelim_x=0.185, lumi_x=0.96, cms_text_size_mult=1.92
            )

        # Section labels (combined plot only)
        sect_objs = []
        if has_sect:
            dw  = 1.0 - left_m - right_m
            slt = ROOT.TLatex(); slt.SetNDC(True)
            slt.SetTextFont(62); slt.SetTextSize(0.065); slt.SetTextAlign(22)
            for sl in deco["section_labels"]:
                cx    = (sl["start"] + sl["end"]) / 2.0
                x_ndc = left_m + (cx / n) * dw
                slt.DrawLatex(x_ndc, 0.88, sl["text"])
            sect_objs.append(slt)

        grp_y    = 0.72 if has_sect else 0.77
        grp_objs = self._draw_group_labels(pad1, deco, left_m, right_m, grp_y)

        # ── ratio pad ─────────────────────────────────────────────────────────
        pad2.cd()
        h_ratio.SetMaximum(1.99); h_ratio.SetMinimum(0.0)
        h_ratio.GetYaxis().SetTitle("")
        h_ratio.GetYaxis().SetLabelSize(0.15); h_ratio.GetYaxis().SetNdivisions(504)
        h_ratio.GetYaxis().SetTickLength(0.015)
        h_ratio.GetXaxis().SetLabelSize(0); h_ratio.GetXaxis().SetTickLength(0.015)
        h_ratio.GetXaxis().SetNdivisions(n, 0, 0, False)
        h_ratio.SetMarkerStyle(20); h_ratio.SetMarkerSize(1.0)
        h_ratio.SetLineColor(ROOT.kBlack); h_ratio.SetStats(0)
        h_ratio.Draw("PE"); h_rband.Draw("E2 SAME"); h_ratio.Draw("PE SAME")

        unity = ROOT.TLine(0, 1, n, 1)
        unity.SetLineColor(ROOT.kRed); unity.SetLineWidth(2); unity.SetLineStyle(2)
        unity.Draw()

        rtitle = ROOT.TLatex(); rtitle.SetNDC(True)
        rtitle.SetTextFont(42); rtitle.SetTextSize(0.16)
        rtitle.SetTextAlign(22); rtitle.SetTextAngle(90)
        rtitle.DrawLatex(0.025, 0.65, "Data / Fit")

        sg_objs = self._draw_sub_group_labels(pad2, deco, left_m, right_m) if has_sub else []
        abcd_y_off = 0.07 if deco.get("section_separator") is not None else 0.0
        bl_objs = self._draw_bin_labels(pad2, deco, left_m, right_m, has_sub, abcd_y_off)

        sep_objs = self._draw_separators(canvas, deco, left_m, right_m, name)

        canvas.Modified(); canvas.Update()
        canvas._keep = [h_bkg, h_sr, h_bkg_err, g_data, h_ratio, h_rband, unity, leg,
                        rtitle, grp_objs, sg_objs, bl_objs, sep_objs, sect_objs, rp_objs]
        return canvas

    # ── Canvas drawing: side-by-side ──────────────────────────────────────────

    def _draw_sidebyside_canvas(self,
                                 a_bkg, a_berr, a_data, a_eyl, a_eyh,
                                 b_bkg, b_berr, b_data, b_eyl, b_eyh,
                                 a_deco, b_deco, name, title):
        split    = 0.30
        mid      = 0.50
        lm_l, lm_r = 0.14, 0.08
        rm       = 0.04

        canvas = ROOT.TCanvas(f"c_{name}", title, 1400, 700)
        canvas.SetFillColor(0)

        pML = ROOT.TPad(f"pML_{name}", "", 0,   split, mid, 1)
        pMR = ROOT.TPad(f"pMR_{name}", "", mid, split, 1,   1)
        pRL = ROOT.TPad(f"pRL_{name}", "", 0,   0,     mid, split)
        pRR = ROOT.TPad(f"pRR_{name}", "", mid, 0,     1,   split)

        bot_l = a_deco.get("bottom_margin", 0.38)
        bot_r = b_deco.get("bottom_margin", 0.38)

        for pad, lm, bm, log in [
            (pML, lm_l, 0.02, True),  (pMR, lm_r, 0.02, True),
            (pRL, lm_l, bot_l, False), (pRR, lm_r, bot_r, False),
        ]:
            pad.SetLeftMargin(lm); pad.SetRightMargin(rm)
            pad.SetTopMargin(0.13 if log else 0.02); pad.SetBottomMargin(bm)
            pad.SetTicks(1, 1)
            if log:
                pad.SetLogy(True); pad.SetGridx(True)
            else:
                pad.SetGridx(True); pad.SetGridy(True)
            pad.Draw()

        keep = []

        for side, (bkg, berr, data, eyl, eyh, deco, pm, pr, lm, is_r) in enumerate([
            (a_bkg, a_berr, a_data, a_eyl, a_eyh, a_deco, pML, pRL, lm_l, False),
            (b_bkg, b_berr, b_data, b_eyl, b_eyh, b_deco, pMR, pRR, lm_r, True),
        ]):
            nb = deco["n_bins"]
            has_sub = bool(deco.get("sub_group_labels"))

            h_bkg = ROOT.TH1F(f"hb_{name}_{side}", "", nb, 0, nb)
            h_bkg.SetDirectory(0); h_bkg.Sumw2()
            for i in range(nb):
                h_bkg.SetBinContent(i + 1, bkg[i]); h_bkg.SetBinError(i + 1, berr[i])
            h_bkg.SetFillColor(ROOT.kAzure - 9)
            h_bkg.SetLineColor(ROOT.kBlue + 1); h_bkg.SetLineWidth(2); h_bkg.SetStats(0)

            h_bkg_err = h_bkg.Clone(f"hberr_{name}_{side}")
            h_bkg_err.SetDirectory(0)
            h_bkg_err.SetFillColor(ROOT.kGray + 1)
            h_bkg_err.SetFillStyle(1001)
            h_bkg_err.SetLineColor(ROOT.kGray + 2)
            h_bkg_err.SetLineWidth(1)

            g_data = ROOT.TGraphAsymmErrors(nb)
            for i in range(nb):
                g_data.SetPoint(i, i + 0.5, data[i])
                g_data.SetPointError(i, 0, 0, eyl[i], eyh[i])
            g_data.SetMarkerStyle(20); g_data.SetMarkerSize(1.2)
            g_data.SetMarkerColor(ROOT.kBlack); g_data.SetLineColor(ROOT.kBlack)
            g_data.SetLineWidth(2)

            h_ratio = ROOT.TH1F(f"hr_{name}_{side}", "", nb, 0, nb)
            h_rband = ROOT.TH1F(f"hrb_{name}_{side}", "", nb, 0, nb)
            h_ratio.SetDirectory(0); h_rband.SetDirectory(0)
            for i in range(nb):
                b = bkg[i]
                h_ratio.SetBinContent(i + 1, data[i] / b if b > 0 else 0)
                h_ratio.SetBinError(i + 1,   max(eyl[i], eyh[i]) / b if b > 0 else 0)
                h_rband.SetBinContent(i + 1, 1.0)
                h_rband.SetBinError(i + 1,   berr[i] / b if b > 0 else 0)
            h_rband.SetFillColor(ROOT.kGray + 1); h_rband.SetFillStyle(3345)
            h_rband.SetMarkerSize(0); h_rband.SetLineColor(0)

            # Main pad
            pm.cd()
            pv = np.concatenate([bkg, data]); pv = pv[pv > 0]
            min_v = max(0.5, 0.3 * pv.min()) if pv.size else 0.5
            max_v = max(float((bkg + berr).max()), float((data + eyh).max()))
            h_bkg.SetMinimum(min_v); h_bkg.SetMaximum(max_v * 10.0)
            h_bkg.GetYaxis().SetLabelSize(0.08)
            if not is_r:
                h_bkg.GetYaxis().SetTitle("Events / bin")
                h_bkg.GetYaxis().SetTitleSize(0.09)
                h_bkg.GetYaxis().SetTitleOffset(0.55)
                h_bkg.GetYaxis().CenterTitle(True)
            h_bkg.GetYaxis().SetTickLength(0.015)
            h_bkg.GetXaxis().SetLabelSize(0); h_bkg.GetXaxis().SetTickLength(0.015)
            h_bkg.GetXaxis().SetNdivisions(nb, 0, 0, False)
            h_bkg.Draw("HIST")
            h_bkg_err.Draw("E2 SAME")
            g_data.Draw("PZ SAME")

            if not is_r:
                leg = ROOT.TLegend(0.57, 0.47, 0.98, 0.74)
                leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.07)
                leg.AddEntry(g_data,    "Data",         "lp")
                leg.AddEntry(h_bkg,     f"{title} bkg", "F")
                leg.Draw(); keep.append(leg)

            grp = self._draw_group_labels(pm, deco, lm, rm, 0.83)

            # Ratio pad
            pr.cd()
            h_ratio.SetMaximum(1.99); h_ratio.SetMinimum(0.0)
            h_ratio.GetYaxis().SetTitle("")
            h_ratio.GetYaxis().SetLabelSize(0.17); h_ratio.GetYaxis().SetNdivisions(504)
            h_ratio.GetYaxis().SetTickLength(0.015)
            h_ratio.GetXaxis().SetLabelSize(0); h_ratio.GetXaxis().SetTickLength(0.015)
            h_ratio.GetXaxis().SetNdivisions(nb, 0, 0, False)
            h_ratio.SetMarkerStyle(20); h_ratio.SetMarkerSize(1.0)
            h_ratio.SetLineColor(ROOT.kBlack); h_ratio.SetStats(0)
            h_ratio.Draw("PE"); h_rband.Draw("E2 SAME"); h_ratio.Draw("PE SAME")

            unity = ROOT.TLine(0, 1, nb, 1)
            unity.SetLineColor(ROOT.kRed); unity.SetLineWidth(2); unity.SetLineStyle(2)
            unity.Draw()

            if not is_r:
                rtitle = ROOT.TLatex(); rtitle.SetNDC(True)
                rtitle.SetTextFont(42); rtitle.SetTextSize(0.19)
                rtitle.SetTextAlign(22); rtitle.SetTextAngle(90)
                rtitle.DrawLatex(0.048, 0.50, "Data / Fit")
                keep.append(rtitle)

            sg = self._draw_sub_group_labels(pr, deco, lm, rm) if has_sub else []
            bl = self._draw_bin_labels(pr, deco, lm, rm, has_sub)

            # Sub-separators within this channel (MsRs only)
            sub_lines = []
            if deco.get("sub_sep_bins"):
                dw = 1.0 - lm - rm
                for sb in deco["sub_sep_bins"]:
                    x = lm + (sb / nb) * dw
                    for pad in [pm, pr]:
                        pad.cd()
                        sl = ROOT.TLine(); sl.SetNDC(True)
                        sl.SetLineColor(ROOT.kGray + 1)
                        sl.SetLineWidth(1); sl.SetLineStyle(3)
                        sl.DrawLine(x, 0.04, x, 0.95)
                        sub_lines.append(sl)

            keep.extend([h_bkg, h_bkg_err, g_data, h_ratio, h_rband, unity, grp, sg, bl, sub_lines])

        # CMS labels on full-canvas overlay
        canvas.cd()
        ov = ROOT.TPad(f"ov_{name}", "", 0, split, 1, 1)
        ov.SetFillStyle(0); ov.SetFrameFillStyle(0)
        ov.SetBorderSize(0); ov.SetBorderMode(0); ov.SetMargin(0, 0, 0, 0)
        ov.SetBit(ROOT.kCannotPick); ov.Draw(); ov.cd()
        self.style.draw_cms_labels(
            cms_x=0.05, cms_y=0.93, prelim_str="Preliminary",
            prelim_x=0.12, lumi_x=0.98, cms_text_size_mult=1.85
        )
        keep.append(ov)

        canvas.Modified(); canvas.Update()
        canvas._keep = keep
        return canvas

    # ── Label helpers ─────────────────────────────────────────────────────────

    def _draw_group_labels(self, pad, deco, left_m, right_m, y_pos=0.6):
        pad.cd()
        n, dw = deco["n_bins"], 1.0 - left_m - right_m
        lt = ROOT.TLatex(); lt.SetNDC(True)
        lt.SetTextFont(42); lt.SetTextSize(0.062); lt.SetTextAlign(22)
        gap = 0.035  # half-gap between two stacked lines
        for g in deco["group_labels"]:
            cx   = (g["start"] + g["end"]) / 2.0
            x    = left_m + (cx / n) * dw
            parts = g["text"].split(", ", 1)
            if len(parts) == 2:
                lt.DrawLatex(x, y_pos + gap, parts[0])
                lt.DrawLatex(x, y_pos - gap, parts[1])
            else:
                lt.DrawLatex(x, y_pos, g["text"])
        return [lt]

    def _draw_sub_group_labels(self, pad, deco, left_m, right_m, y_pos=0.43):
        pad.cd()
        if not deco.get("sub_group_labels"):
            return []
        n, dw = deco["n_bins"], 1.0 - left_m - right_m
        objs  = []

        lt = ROOT.TLatex(); lt.SetNDC(True)
        lt.SetTextFont(42); lt.SetTextSize(0.12); lt.SetTextAlign(22)
        for sg in deco["sub_group_labels"]:
            cx = (sg["start"] + sg["end"]) / 2.0
            lt.DrawLatex(left_m + (cx / n) * dw, y_pos, sg["text"])
        objs.append(lt)

        sg_title = deco.get("sub_group_axis_title", "")
        if sg_title:
            tlt = ROOT.TLatex(); tlt.SetNDC(True)
            tlt.SetTextFont(42); tlt.SetTextSize(0.12); tlt.SetTextAlign(32)
            tlt.DrawLatex(left_m + 0.005, y_pos, sg_title)
            objs.append(tlt)

        return objs

    def _draw_bin_labels(self, pad, deco, left_m, right_m, has_sub_groups=False, abcd_y_offset=0.0):
        pad.cd()
        n, dw = deco["n_bins"], 1.0 - left_m - right_m
        y_bin   = 0.31 if has_sub_groups else 0.32
        y_title = 0.09 if has_sub_groups else 0.12
        sect_sep = deco.get("section_separator")

        is_risr = deco.get("bin_scheme") == "risr"
        lt = ROOT.TLatex(); lt.SetNDC(True)
        #lt.SetTextFont(42); lt.SetTextSize(0.13)
	    #CHANGE BIN LABELS SIZE HERE!!!!
        lt.SetTextFont(42); lt.SetTextSize(0.1)
        if is_risr:
            lt.SetTextAngle(-20)
            lt.SetTextAlign(12)  # left end at bin left-edge, text slopes down-right into margin
        else:
            lt.SetTextAlign(22)
        for i, label in enumerate(deco["bin_labels"]):
            if is_risr:
                x = left_m + (i / n) * dw          # left edge of bin
            else:
                x = left_m + ((i + 0.5) / n) * dw  # centre of bin
            y = y_bin + (0.04 if is_risr else -0.05) + (abcd_y_offset if sect_sep is not None and i < sect_sep else 0.0)
            lt.DrawLatex(x, y, label)

        objs = [lt]
        xt = deco.get("x_axis_title", "")
        if xt:
            tlt = ROOT.TLatex(); tlt.SetNDC(True)
            tlt.SetTextFont(42); tlt.SetTextSize(0.17); tlt.SetTextAlign(22)
            tlt.DrawLatex(left_m + 0.5 * dw, y_title, xt)
            objs.append(tlt)
        return objs

    def _draw_separators(self, canvas, deco, left_m, right_m, name):
        n, dw = deco["n_bins"], 1.0 - left_m - right_m

        ov = ROOT.TPad(f"sep_{name}", "", 0, 0, 1, 1)
        ov.SetFillStyle(0); ov.SetFrameFillStyle(0)
        ov.SetBorderSize(0); ov.SetBorderMode(0); ov.SetMargin(0, 0, 0, 0)
        ov.SetBit(ROOT.kCannotPick)
        canvas.cd(); ov.Draw(); ov.cd()

        objs = [ov]

        y2 = 0.91

        for sb in deco.get("sub_sep_bins", []):
            x = left_m + (sb / n) * dw
            ln = ROOT.TLine(); ln.SetNDC(True)
            ln.SetLineColor(ROOT.kGray + 1); ln.SetLineWidth(1); ln.SetLineStyle(3)
            ln.DrawLine(x, 0.08, x, y2); objs.append(ln)

        for sb in deco.get("separator_bins", []):
            x = left_m + (sb / n) * dw
            ln = ROOT.TLine(); ln.SetNDC(True)
            ln.SetLineColor(ROOT.kGray + 2); ln.SetLineWidth(2); ln.SetLineStyle(2)
            ln.DrawLine(x, 0.08, x, y2); objs.append(ln)

        ss = deco.get("section_separator")
        if ss is not None:
            x = left_m + (ss / n) * dw
            ln = ROOT.TLine(); ln.SetNDC(True)
            ln.SetLineColor(ROOT.kBlack); ln.SetLineWidth(3); ln.SetLineStyle(1)
            ln.DrawLine(x, 0.08, x, 0.91); objs.append(ln)

        canvas.Modified(); canvas.Update()
        return objs

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _ch_short(ch):
        return ch.lower().replace("cr", "").replace("_", "")

    def _flush(self, plots, output_prefix, fmt):
        """
        Save all collected (canvas_pre, canvas_post, name) tuples.

        ROOT format: one file at {output_prefix}.root containing every canvas,
                     named '{name}_prefit' / '{name}_postfit'.
        Other formats: one file per canvas at {output_prefix}_{name}_{tag}.{fmt}.
        """
        if fmt == "root":
            path = f"{output_prefix}.root"
            fo   = ROOT.TFile(path, "RECREATE")
            for c_pre, c_post, name in plots:
                c_pre.SetName(f"{name}_prefit")
                c_post.SetName(f"{name}_postfit")
                c_pre.Write()
                c_post.Write()
            fo.Close()
            print(f"Saved: {path}  ({len(plots)*2} canvases)")
        else:
            for c_pre, c_post, name in plots:
                for c, tag in [(c_pre, "prefit"), (c_post, "postfit")]:
                    path = f"{output_prefix}_{name}_{tag}.{fmt}"
                    c.SaveAs(path)
                    print(f"Saved: {path}")
