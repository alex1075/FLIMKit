#!/usr/bin/env python
"""CLI entry-point for interactive phasor FLIM analysis.

Run from the terminal::

    python phasor_cli.py                     # guided prompts
    python phasor_cli.py --ptu TTTT.ptu      # skip first prompt
    python phasor_cli.py --session prev.npz  # resume saved session
"""

import warnings
warnings.filterwarnings("ignore")

from flimkit.phasor_launcher import launch_phasor, phasor_inquire


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Interactive Phasor FLIM Analysis — guided mode by default"
    )
    ap.add_argument("--ptu", default=None,
                    help="Path to a .ptu file (skips the first prompt)")
    ap.add_argument("--irf", default=None,
                    help="Path to the IRF calibration Excel file")
    ap.add_argument("--session", default=None,
                    help="Resume a previously saved .npz session")
    args = ap.parse_args()

    # If any explicit paths were given, pass them straight through
    if args.ptu or args.session:
        launch_phasor(
            ptu_path=args.ptu,
            irf_path=args.irf,
            session_path=args.session,
        )
    else:
        # Fully guided inquirer flow
        phasor_inquire()


if __name__ == "__main__":
    main()
