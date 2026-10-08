import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_evidence import summarize, verify
from verify_model import HeaderReader, header


def string(value):
    raw = value.encode()
    return struct.pack('<Q',len(raw))+raw


class PublicationTests(unittest.TestCase):
    def test_scores_recalculate_from_persisted_ids(self):
        self.assertTrue(verify()['passed'])

    def test_duplicate_scores_are_rejected(self):
        rows = [{'id':'same','correct':True,'truncated':False}]*2
        with self.assertRaises(ValueError):
            summarize(rows)

    def test_gguf_header_counts_actual_tensor_shapes(self):
        data = b'GGUF'+struct.pack('<IQQ',3,1,2)
        data += string('general.architecture')+struct.pack('<I',8)+string('qwen35')
        # A tokenizer string array must be skipped without loading weights.
        data += string('tokenizer.ggml.tokens')+struct.pack('<IIQ',9,8,2)+string('a')+string('b')
        data += string('test.weight')+struct.pack('<IQQIQ',2,128,4,142,0)
        data += b'\0'*136
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fixture.gguf';path.write_bytes(data)
            result = header(path)
        self.assertEqual(result['metadata'],{'general.architecture':'qwen35'})
        self.assertEqual(result['tensor_types'],{142:1})
        self.assertEqual(result['parameters_by_type'],{142:512})

    def test_truncated_gguf_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.gguf';path.write_bytes(b'GGUF'+struct.pack('<IQQ',3,1,0))
            with self.assertRaises(ValueError):
                header(path)

    def test_unknown_value_type_is_rejected(self):
        with self.assertRaises(ValueError):
            HeaderReader(io.BytesIO(b'\0'*8)).value(999)


if __name__ == '__main__':
    unittest.main()
