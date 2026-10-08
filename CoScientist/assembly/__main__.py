"""Validate system.yaml and print the build plan:  python -m CoScientist.assembly"""
from CoScientist.assembly.assembler import load_config_cli

load_config_cli()
