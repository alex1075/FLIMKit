#!/usr/bin/env python
from pathlib import Path
from code.interactive import *
import inquirer



def main():
    print("Welcome to the FLIM data processing tool!")
    
    questions = [
        inquirer.List(
            'process_option',
            message="Choose a processing option",
            choices=[
                'FLIM FIT a single FOV',
                'Reconstruct a FOV and FLIM FIT',
                'Just stitch multiple tiles together',
                'Exit'
            ]
        )
    ]
    answers = inquirer.prompt(questions)
    
    if answers['process_option'] == 'FLIM FIT a single FOV':
        print("FLIM FITting a single FOV...")
        single_FOV_flim_fit(interactive=True)
        
        
    elif answers['process_option'] == 'Reconstruct a FOV and FLIM FIT':
        print("Reconstructing a FOV and FLIM FITting...")
        print("This option is not yet implemented. Please check back later.")
        
    elif answers['process_option'] == 'Just stitch multiple tiles together':
        print("Stitching multiple tiles together...")
        print("This option is not yet implemented. Please check back later.")
        
    else:
        print("Exiting. Goodbye!")
        return
    
if __name__ == "__main__":
    main()    