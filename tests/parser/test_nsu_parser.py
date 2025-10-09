from parser.nsu_parser import parse_nsu_faculty


class TestNsuParser:
    def test_parse_nsu_fit(self):
        data = parse_nsu_faculty("information-technologies/")

        assert len(data) == 5
        assert data["total_blocks"] == 87
