""" Usage:
    <file-name> --in=IN_FILE --out=OUT_FILE [--debug]
"""
import logging
import pdb
from pprint import pprint
from pprint import pformat
from docopt import docopt
from collections import defaultdict
from operator import itemgetter
from tqdm import tqdm
import os

def read_sents_from_file(text_fn):
    """
    Read new-line separated files.
    """
    return iter([sent.strip().replace(" ", "")
                 for sent in open(text_fn, encoding = "utf8")
                 if sent.strip()])

if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    inp_fn = args["--in"]
    out_fn = args["--out"]
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    base_text_dir = "./CADEC_v2/cadec/text/"
    cur_sents = None
    cur_sent = None
    with open(out_fn, "w", encoding = "utf8") as fout:
        for line in tqdm(open(inp_fn, encoding = "utf8")):
            line = line.strip()
            data = line.split()
            if len(data) == 1:
                # Read a new file
                cur_file = data[0]
                cur_sents = read_sents_from_file(os.path.join(base_text_dir, f"{cur_file}.txt"))
                cur_sent = next(cur_sents)
                fout.write(f"{line}\n")
            elif len(data) == 0:
                # Read next sentence
                assert (not cur_sent)
                try:
                    cur_sent = next(cur_sents)
                except StopIteration:
                    cur_sent = None
                fout.write("\n")
            elif len(data) == 5:
                raise AssertionError
            else:
                # Replace cur token with the word from the file
                cur_word_len = int(data[0])
                cur_word = cur_sent[: cur_word_len]
                cur_sent = cur_sent[cur_word_len: ]
                data[0] = cur_word
                fout.write("\t".join(data) + "\n")

    logging.info("DONE")