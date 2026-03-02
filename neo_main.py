#!/usr/bin/env python
from pathlib import Path

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
        
        
    elif answers['process_option'] == 'Reconstruct a FOV and FLIM FIT':
        print("Reconstructing a FOV and FLIM FITting...")
        # Call function to reconstruct a FOV and FLIM FIT (to be implemented)
        
    elif answers['process_option'] == 'Just stitch multiple tiles together':
        print("Stitching multiple tiles together...")
        # Call function to stitch tiles (to be implemented)
        
    else:
        print("Exiting. Goodbye!")
        return