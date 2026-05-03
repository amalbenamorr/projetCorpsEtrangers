"""
main.py — v2
─────────────
Entry point for ELMAZRAA Visual Forensics Agent.

Usage:
  python main.py --image path/to/image.jpg
  python main.py --image path/to/image.jpg --station "Station-3"
  python main.py --demo       (runs on first image in data/)
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="ELMAZRAA Visual Forensics Agent — Foreign Object Detection"
    )
    parser.add_argument("--image",   type=str, default=None,
                        help="Path to image file")
    parser.add_argument("--station", type=str, default="Station-1",
                        help="Production line station")
    parser.add_argument("--demo",    action="store_true",
                        help="Run demo on first image in data/")
    parser.add_argument("--sensitivity", type=str, default="auto",
                        choices=["auto", "high", "low"],
                        help="Scanner sensitivity (default: auto)")
    args = parser.parse_args()

    # Resolve image path
    image_path = None
    if args.demo:
        data_dir = Path("data")
        images   = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
        if not images:
            print("❌ No images found in data/ directory")
            sys.exit(1)
        image_path = images[0]
        print(f"🎬 Demo mode — using: {image_path}")
    elif args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)

    # ── Run inspection ────────────────────────────────────────────────────
    from agents.master_agent import run_inspection

    print(f"\n{'═'*60}")
    print(f"  ELMAZRAA — Visual Forensics Agent v5")
    print(f"  Image  : {image_path}")
    print(f"  Station: {args.station}")
    print(f"{'═'*60}\n")

    result = run_inspection(
        image_input=str(image_path),
        station=args.station
    )

    # ── Print final result ─────────────────────────────────────────────────
    safe_result = {
        k: v for k, v in result.items()
        if k not in ("full_image", "heatmap", "zones", "alert", "memory")
    }

    print("\n" + "═" * 60)
    print("FINAL INSPECTION REPORT")
    print("═" * 60)
    print(json.dumps(safe_result, indent=2, default=str))

    alert = result.get("alert", {})
    if alert.get("gif_path"):
        print(f"\n📁 GIF saved: {alert['gif_path']}")
    if alert.get("report_path"):
        print(f"📄 Report  : {alert['report_path']}")

    memory = result.get("memory", {})
    if memory.get("success"):
        print(f"🧠 Memory  : case {memory.get('case_id', '')[:8]}... written")
        print(f"            Total cases: {memory.get('total_in_memory', '?')}")

    print("\n" + "═" * 60)
    return result


if __name__ == "__main__":
    main()