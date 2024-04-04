from main import parse, train, report
from src.DRHGCN.model import DRHGCN
import os

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parse(print_help=False)
    train(args, DRHGCN)
    # report("runs")
