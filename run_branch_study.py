from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
if __name__=='__main__':
    from paper_reproduction.runner import main
    main()
