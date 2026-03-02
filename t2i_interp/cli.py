import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: t2i <command> [args...]")
        print("Available commands:")
        print("  steer      Run steering sweeps")
        print("  localise   Run localisation ablation sweeps")
        print("  sae        Run SAE tools")
        print("  stitch     Run patching/stitching tools")
        sys.exit(1)
        
    command = sys.argv[1]
    
    # Pop the command from argv so hydra doesnt accidentally consume it as an unparsed override
    sys.argv.pop(1)
    
    if command == "steer":
        from t2i_interp.scripts.run_steer import main as cmd_main
    elif command == "localise":
        from t2i_interp.scripts.run_localisation import main as cmd_main
    elif command == "sae":
        from t2i_interp.scripts.run_sae import main as cmd_main
    elif command == "stitch":
        from t2i_interp.scripts.run_stitch import main as cmd_main
    else:
        print(f"Unknown command: '{command}'")
        print("Available commands: steer, localise, sae, stitch")
        sys.exit(1)
        
    # Launch sub-script
    cmd_main()

if __name__ == "__main__":
    main()
