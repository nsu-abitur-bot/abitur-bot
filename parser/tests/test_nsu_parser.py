import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from parser.nsu_parser import parse_nsu_faculty

class TestNSUParser:
    def test_parse_fit(self):
        result = parse_nsu_faculty("information-technologies/")