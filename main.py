#!/usr/bin/env python
from pathlib import Path
from pyflim.interactive import *
import inquirer
from pyflim._version import __version__, roadmap
from pyflim.utils.fancy import display_banner, flim_fitting_banner, banner_goodbye



def main(fast = False):
    if fast == False:
        display_banner()
    print("Welcome to the FLIM data processing tool!")
    
    questions = [
        inquirer.List(
            'process_option',
            message="Choose a processing option",
            choices=[
                'FLIM FIT a single FOV',
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
    main(False)    