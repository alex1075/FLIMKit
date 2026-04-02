#!/usr/bin/env python
# ── matplotlib font-cache fast-path (must come before ANY matplotlib import) ──
import os, sys
if getattr(sys, 'frozen', False):
    # Running as a PyInstaller bundle — point at the bundled mpl-cache/
    _mpl_cache = os.path.join(sys._MEIPASS, 'mpl-cache')
else:
    # Running from source — use a local mpl-cache/ next to this file
    _mpl_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mpl-cache')
os.makedirs(_mpl_cache, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', _mpl_cache)
# ─────────────────────────────────────────────────────────────────────────────
import argparse
from pathlib import Path



def main(fast=False, cli=False):
    if not cli:
        from flimkit.UI.gui import launch_gui
        launch_gui()
        return
    
    # Only import interactive components when needed (not GUI mode)
    from flimkit.interactive import single_FOV_flim_fit, stitch_and_fit, stitch_tiles
    import inquirer
    from flimkit._version import __version__, roadmap
    from flimkit.utils.fancy import display_banner, flim_fitting_banner, banner_goodbye
    
    if fast == False:
        display_banner()
    print("Welcome to the FLIM data processing tool!")
    
    questions = [
        inquirer.List(
            'process_option',
            message="Choose a processing option",
            choices=[
                'FLIM FIT a single FOV',
                'Phasor analysis',
                'Reconstruct a FOV and FLIM FIT',
                'Just stitch multiple tiles together',
                'About',
                'Exit'
            ]
        )
    ]
    answers = inquirer.prompt(questions)
    
    if answers['process_option'] == 'FLIM FIT a single FOV':
        if fast == False:
            flim_fitting_banner()
        print("FLIM FITting a single FOV...")
        single_FOV_flim_fit(interactive=True)
        
    elif answers['process_option'] == 'Phasor analysis':
        from flimkit.phasor_launcher import phasor_inquire
        phasor_inquire()
        
    elif answers['process_option'] == 'Reconstruct a FOV and FLIM FIT':
        print("Reconstructing a FOV and FLIM FITting...")
        stitch_and_fit(interactive=True)
        
    elif answers['process_option'] == 'Just stitch multiple tiles together':
        print("Stitching multiple tiles together...")
        stitch_tiles(interactive=True)
    elif answers['process_option'] == 'About':
        print('Current version: ' + __version__)
        print(roadmap)
        return    
    else:
        banner_goodbye()
        return
    
if __name__ == "__main__":
    # Required for PyInstaller + multiprocessing on macOS/Windows.
    # Must be the first call inside __main__; harmless when not frozen.
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="FLIMKit — FLIM data processing toolkit")
    parser.add_argument('--cli', action='store_true', help='=Run in CLI mode')
    parser.add_argument('--fast', action='store_true', help='Skip banner display')
    args = parser.parse_args()
    
    main(fast=args.fast, cli=args.cli)    