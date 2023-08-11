from create_graph import get_pred_res_2
from neo4j_connect import OperationNeo4j
import argparse


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sens', type=str,
                        default="机床导轨副的磨损与工作的连续性、负荷特性、工作条件、导轨的材质和结构等有关。")
    args = parser.parse_args()
    sens = args.sens
    res = get_pred_res_2(sens)
    op_neo4j = OperationNeo4j()
    op_neo4j.graph.run(res)
