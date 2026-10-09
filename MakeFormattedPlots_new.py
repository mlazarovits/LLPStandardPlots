#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import ROOT
from ROOT import TFile

from PlotFormatter import PlotFormatter, PlotFormatHelper


# -----------------------------------------------------------------------------
# Observable configuration
# -----------------------------------------------------------------------------
# "dimension" describes the observable itself.  It does NOT determine whether
# a 1D plot is an overlay or a Data/MC stack; those are separate plot modes.
OBS_CONFIG = {
    "Ms": {
        "dimension": "1d",
        "xlabel": "M_{S} [TeV]",
        "xmin": 0.0,
        "xmax": 3.0,
    },
    "Rs": {
        "dimension": "1d",
        "xlabel": "R_{S}",
        "xmin": 0.0,
        "xmax": 1.0,
    },
    "timesig": {
        "dimension": "1d",
        "xlabel": "Time significance",
        "xmin": 0.0,
        "xmax": 10.0,
    },
    "MsRs": {
        "dimension": "2d",
        "xlabel": "M_{S} [TeV]",
        "ylabel": "R_{S}",
        "xmin": 0.0,
        "xmax": 7.0,
        "ymin": 0.0,
        "ymax": 1.0,
    },
    "PtISRRISR": {
        "dimension": "2d",
        "xlabel": "p^{ISR}_{T} [GeV]",
        "ylabel": "R_{ISR}",
        "xmin": 50.0,
        "xmax": 3000.0,
        "ymin": 0.0,
        "ymax": 1.0,
    },
}


# -----------------------------------------------------------------------------
# Parsing / naming helpers
# -----------------------------------------------------------------------------
def parse_plot_spec(spec):
    """Parse PROCESS:REGION:OBS."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --plot entry '{spec}'. Expected PROCESS:REGION:OBS."
        )

    process, region, obs = parts
    if not process or not region or not obs:
        raise ValueError(
            f"Invalid --plot entry '{spec}'. PROCESS, REGION, and OBS are required."
        )

    if obs not in OBS_CONFIG:
        raise ValueError(
            f"Unknown observable '{obs}'. Available observables: "
            + ", ".join(OBS_CONFIG)
        )

    return process, region, obs


def parse_plot_group(spec):
    """
    Parse a generic overlay/2D --plot argument.

    Example:
        SMS1:SR:Ms,SMS2:SR:Ms,METPD:SR:Ms

    All entries in a group must refer to the same observable and dimension.
    A 2D group must contain exactly one histogram.
    """
    entries = [parse_plot_spec(item.strip()) for item in spec.split(",") if item.strip()]

    if not entries:
        raise ValueError(f"Empty --plot argument: '{spec}'")

    observables = {entry[2] for entry in entries}
    if len(observables) != 1:
        raise ValueError(
            f"All entries in one --plot group must use the same observable: '{spec}'"
        )

    obs = entries[0][2]
    dimension = OBS_CONFIG[obs]["dimension"]

    if dimension == "2d" and len(entries) != 1:
        raise ValueError(
            f"2D observable '{obs}' accepts exactly one PROCESS:REGION:OBS entry "
            "per --plot argument."
        )

    return entries


def parse_datamc_spec(spec):
    """
    Parse DATA:MC1,MC2,...:REGION:OBS.
    """
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"Invalid --datamc entry '{spec}'. Expected DATA:MC1,MC2,...:REGION:OBS."
        )

    data_process, mc_string, region, obs = parts
    mc_processes = [p.strip() for p in mc_string.split(",") if p.strip()]

    if not data_process or not mc_processes or not region or not obs:
        raise ValueError(
            f"Invalid --datamc entry '{spec}'. DATA, MC processes, REGION, and OBS are required."
        )

    if obs not in OBS_CONFIG:
        raise ValueError(
            f"Unknown observable '{obs}'. Available observables: "
            + ", ".join(OBS_CONFIG)
        )

    if OBS_CONFIG[obs]["dimension"] != "1d":
        raise ValueError(
            f"--datamc only supports 1D observables; '{obs}' is 2D."
        )

    return data_process, mc_processes, region, obs


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def is_data_process(process):
    """Match the data-process convention used by PlotFormatter."""
    return "PD" in process or process.startswith("MET")


def clone_hist(hist, histtype):
    """
    Clone a histogram before applying plot-specific transformations.

    This prevents normalization, sorting, GeV->TeV conversion, etc. in one
    plot from modifying the histogram used by another plot.
    """
    base_name = hist.GetName()
    suffix = f"_{histtype}"
    clone_name = base_name if base_name.endswith(suffix) else base_name + suffix

    clone = hist.Clone(clone_name)
    clone.SetDirectory(0)
    return clone


# -----------------------------------------------------------------------------
# Histogram retrieval
# -----------------------------------------------------------------------------
def get_hist(helper, process, region, obs, histtype):
    hists = helper.GetHists(
        process=process,
        obs=obs,
        region=region,
        histtype=None,
    )

    if not hists:
        return None

    return clone_hist(hists[0], histtype)


# -----------------------------------------------------------------------------
# Styling for generic 1D overlays
# -----------------------------------------------------------------------------
def get_overlay_style(helper, process, index):
    """
    Return (ROOT color, line style) for an overlay histogram.

    Data is black, signals use a cycling signal palette/dashed styles, and
    backgrounds use the same background colors already defined by
    PlotFormatHelper where possible.
    """
    if is_data_process(process):
        return ROOT.kBlack, 1

    if "SMS" in process:
        signal_colors = [
            ROOT.kGreen-9,
            ROOT.kGreen-8,
            ROOT.kGreen-7,
            ROOT.kGreen-6
            #ROOT.kRed + 1,
            #ROOT.kBlue + 1,
            #ROOT.kGreen + 2,
            #ROOT.kMagenta + 1,
            #ROOT.kOrange + 7,
            #ROOT.kCyan + 1,
        ]
        signal_styles = [0]#,2, 3, 4, 5, 7, 9]
        signal_index = index % len(signal_colors)
        return signal_colors[signal_index], signal_styles[0]

    try:
        color = helper._get_background_color_index(process)
    except Exception:
        color = ROOT.kGray + 2 + (index % 5)

    return color, 1


# -----------------------------------------------------------------------------
# Plot construction
# -----------------------------------------------------------------------------
def make_1d_overlay_plot(formatter, helper, entries, normalize=False, logy=False):
    obs = entries[0][2]
    config = OBS_CONFIG[obs]

    hists = []
    processes = []

    for process, region, _ in entries:
        hist = get_hist(helper, process, region, obs, "1d")
        if hist is None:
            print(
                f"WARNING: no histogram found for process={process}, "
                f"region={region}, obs={obs}; skipping it."
            )
            continue

        if obs == "Ms":
            hist = helper.GeVtoTeV(hist)

        hists.append(hist)
        processes.append(process)

    if not hists:
        print(f"WARNING: no histograms found for overlay '{obs}'. Skipping plot.")
        return None

    colors = []
    styles = []
    for i, process in enumerate(processes):
        color, style = get_overlay_style(helper, process, i)
        colors.append(color)
        styles.append(style)

    # MakeLegendLabels provides the region/channel label separately from the
    # individual process labels.  Only draw a global label when there is one
    # unambiguous group label.
    try:
        _, group_labels = helper.MakeLegendLabels([hist.GetName() for hist in hists])
        global_label = group_labels[0] if len(group_labels) == 1 else ""
    except Exception:
        global_label = ""

    canvas_name = "overlay_" + safe_name(
        obs + "_" + "__".join(f"{p}_{r}" for p, r, _ in entries)
    )

    canvas = formatter.format_hists_1d(
        canvas_name,
        hists,
        colors,
        styles,
        config["xlabel"],
        obs,
        normalize=normalize,
        logy=logy,
        mc=not any(is_data_process(p) for p in processes),
        globallabel=global_label,
        fillstyle=0,
    )

    return canvas


def make_2d_plot(formatter, helper, entries, normalize=False):
    process, region, obs = entries[0]
    config = OBS_CONFIG[obs]

    hist = get_hist(helper, process, region, obs, "2d")
    if hist is None:
        print(
            f"WARNING: no histogram found for process={process}, "
            f"region={region}, obs={obs}; skipping plot."
        )
        return None

    # Ms is stored in GeV but displayed in TeV.  For a TH2, GeVtoTeV must be
    # told explicitly to convert the x axis.
    if obs == "MsRs":
        hist = helper.GeVtoTeV(hist, axis=0)

    try:
        _, group_labels = helper.MakeLegendLabels([hist.GetName()])
        print("group_labels",group_labels)
        global_label = group_labels[1] if len(group_labels) == 2 else ""
    except Exception:
        global_label = ""

    canvas_name = "2d_" + safe_name(f"{obs}_{process}_{region}")

    canvas = formatter.format_2d_hist(
        canvas_name,
        hist,
        process,
        config["xlabel"],
        config["xmin"],
        config["xmax"],
        config["ylabel"],
        config["ymin"],
        config["ymax"],
        normalize=normalize,
        globallabel=global_label,
    )

    return canvas


def make_datamc_plot(formatter, helper, data_process, mc_processes, region, obs,
                     normalize=False, logy=False):
    data_hist = get_hist(helper, data_process, region, obs, "stack")
    if data_hist is None:
        print(
            f"WARNING: no data histogram found for process={data_process}, "
            f"region={region}, obs={obs}; skipping Data/MC plot."
        )
        return None

    mc_hists = []
    valid_mc_processes = []
    for process in mc_processes:
        hist = get_hist(helper, process, region, obs, "stack")
        if hist is None:
            print(
                f"WARNING: no MC histogram found for process={process}, "
                f"region={region}, obs={obs}; skipping that MC process."
            )
            continue
        mc_hists.append(hist)
        valid_mc_processes.append(process)

    if not mc_hists:
        print(f"WARNING: no MC histograms found for Data/MC '{obs}'. Skipping plot.")
        return None

    if obs == "Ms":
        data_hist = helper.GeVtoTeV(data_hist)
        mc_hists = [helper.GeVtoTeV(hist) for hist in mc_hists]

    config = OBS_CONFIG[obs]

    # The existing Data/MC stack formatter expects one data histogram followed
    # by the MC histograms and handles the stack/ratio construction itself.
    canvas_name = "datamc_" + safe_name(f"{obs}_{region}")

    canvas = formatter.format_stack_hists_datamc(
        canvas_name,
        data_hist,
        mc_hists,
        config["xlabel"],
        config["xmin"],
        config["xmax"],
        normalize,
        draw_mc_uncertainty=True,
        globallabel=region,
    )

    return canvas


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def save_canvas(canvas, output_dir, image_format, root_file=None):
    if canvas is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{canvas.GetName()}.{image_format}"
    canvas.SaveAs(str(output_path))
    print(f"Saved: {output_path}")

    if root_file is not None:
        root_file.cd()
        canvas.Write()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Make CMS-formatted 1D overlays, Data/MC stacks, and 2D plots."
    )
    parser.add_argument(
        "-i", "--inputFile", required=True,
        help="Input ROOT file containing the histograms."
    )
    parser.add_argument(
        "-o", "--ofileName", required=True,
        help="Output directory for formatted plots."
    )
    parser.add_argument(
        "--plot", action="append", default=[],
        metavar="P:R:OBS[,P:R:OBS...]",
        help=(
            "Generic overlay/2D plot. For 1D, multiple entries in one argument "
            "are overlaid; for 2D, provide one entry. May be repeated."
        ),
    )
    parser.add_argument(
        "--datamc", action="append", default=[],
        metavar="DATA:MC1,MC2:REGION:OBS",
        help="Make a Data/MC stack plot. May be repeated.",
    )
    parser.add_argument(
        "--format", "-f", choices=["png", "pdf"], default="png",
        help="Output image format (default: png)."
    )
    parser.add_argument(
        "--save-root", action="store_true",
        help="Also save the produced canvases into an output ROOT file."
    )
    parser.add_argument(
        "--norm", action="store_true",
        help="Normalize histograms where supported."
    )
    parser.add_argument(
        "--logy", action="store_true",
        help="Use a logarithmic y axis for 1D plots."
    )
    parser.add_argument(
        "--lumi", default=None,
        help="Set luminositu"
    )

    args = parser.parse_args()

    if not args.plot and not args.datamc:
        parser.error("Provide at least one --plot or --datamc argument.")

    # Parse everything before opening the input file so malformed arguments
    # fail immediately.
    try:
        plot_groups = [parse_plot_group(spec) for spec in args.plot]
        datamc_groups = [parse_datamc_spec(spec) for spec in args.datamc]
    except ValueError as exc:
        parser.error(str(exc))

    input_file = TFile.Open(args.inputFile, "READ")
    if not input_file or input_file.IsZombie():
        raise RuntimeError(f"Could not open input ROOT file: {args.inputFile}")

    helper = PlotFormatHelper(input_file)
    formatter = PlotFormatter(helper)
    formatter.SetLumi(args.lumi)
    formatter.SetEnergy(13.6)

    output_dir = Path(args.ofileName)
    root_file = None
    if args.save_root:
        output_dir.mkdir(parents=True, exist_ok=True)
        root_path = output_dir / "formatted_plots.root"
        root_file = TFile.Open(str(root_path), "RECREATE")
        if not root_file or root_file.IsZombie():
            raise RuntimeError(f"Could not create output ROOT file: {root_path}")

    n_plots = 0

    # Generic --plot mode: each argument is one overlay (1D) or one 2D plot.
    for entries in plot_groups:
        obs = entries[0][2]
        dimension = OBS_CONFIG[obs]["dimension"]

        if dimension == "1d":
            canvas = make_1d_overlay_plot(
                formatter,
                helper,
                entries,
                normalize=args.norm,
                logy=args.logy,
            )
        elif dimension == "2d":
            canvas = make_2d_plot(
                formatter,
                helper,
                entries,
                normalize=args.norm,
            )
        else:
            raise RuntimeError(f"Unsupported dimension '{dimension}' for '{obs}'.")

        if canvas is not None:
            save_canvas(canvas, output_dir, args.format, root_file)
            n_plots += 1

    # Data/MC stack mode remains separate from generic overlays.
    for data_process, mc_processes, region, obs in datamc_groups:
        canvas = make_datamc_plot(
            formatter,
            helper,
            data_process,
            mc_processes,
            region,
            obs,
            normalize=args.norm,
            logy=args.logy,
        )

        if canvas is not None:
            save_canvas(canvas, output_dir, args.format, root_file)
            n_plots += 1

    if root_file is not None:
        print("Wrote canvases to",root_file.GetName())
        root_file.Close()

    input_file.Close()

    if n_plots == 0:
        print("No plots were produced.")
        return 1

    print(f"Produced {n_plots} plot(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
