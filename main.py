"""Entry point.

    python main.py                      the production window: main prompt in,
                                        film out (blender_ai_studio/production_gui)
    python main.py --studio             the single-shot studio: one prompt, one
                                        image or clip
    python main.py --cli "a prompt"     the studio, headless
"""

import argparse
import sys

from blender_ai_studio import config, console


def run_cli(prompt, mode, overrides):
    from blender_ai_studio.runner import Runner

    settings = config.load_config()
    settings.update({k: v for k, v in overrides.items() if v is not None})
    if not config.resolve_api_key(settings.get("api_key")):
        print("No NVIDIA API key. Set NVIDIA_API_KEY, or add it to config.json.")
        return 2

    runner = Runner(
        settings,
        on_log=lambda m: print("  " + console.safe(m), flush=True),
        on_progress=lambda f, label: print(
            f"[{f * 100:5.1f}%] " + console.safe(label), flush=True
        ),
    )
    result = runner.run(prompt, mode)
    if result.error:
        print(f"\nFailed: {result.error}")
        return 1
    if result.cancelled:
        print("\nCancelled.")
        return 1
    print(f"\nOutput:  {result.output_path}")
    print(f"Folder:  {result.run_dir}")
    return 0


def main():
    console.configure()
    if config.migrate_secrets():
        print("Moved your API key out of config.json into .env")
    parser = argparse.ArgumentParser(description="Blender AI Studio")
    parser.add_argument("--cli", metavar="PROMPT", help="run without the GUI")
    parser.add_argument("--mode", choices=["image", "animation"], default="image")
    parser.add_argument("--resolution", help='e.g. "1280x720"')
    parser.add_argument("--engine", choices=["EEVEE", "CYCLES"])
    parser.add_argument("--samples", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--studio", action="store_true",
                        help="open the single-shot studio instead of the "
                             "production window")
    args = parser.parse_args()

    overrides = {
        "resolution": args.resolution,
        "engine": args.engine,
        "samples": args.samples,
        "frames": args.frames,
        "fps": args.fps,
        "port": args.port,
    }

    if args.cli:
        return run_cli(args.cli, args.mode, overrides)

    if args.studio:
        from blender_ai_studio.gui import main as gui_main
    else:
        from blender_ai_studio.production_gui import main as gui_main

    gui_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
