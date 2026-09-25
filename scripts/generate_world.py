"""Generate a fake store world.

    python scripts/generate_world.py --preset tiny
    python scripts/generate_world.py --preset small --seed 7
"""

import argparse

from shopper_twin.world import PRESETS, generate_world, get_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="small")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--shoppers", type=int, default=None)
    parser.add_argument("--out", default="data/worlds")
    args = parser.parse_args()

    overrides = {k: v for k, v in {"seed": args.seed, "n_days": args.days, "n_shoppers": args.shoppers}.items()
                 if v is not None}
    generate_world(get_config(args.preset, **overrides), root=args.out)


if __name__ == "__main__":
    main()
