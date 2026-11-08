from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
from part_a import run as run_a
from part_b import run as run_b
from part_c import run as run_c
from part_d import run as run_d

if __name__=="__main__":
    print("PART A"); run_a()
    print("PART B"); run_b()
    print("PART C"); run_c()
    print("PART D"); run_d()
    print("PIPELINE COMPLETE")
