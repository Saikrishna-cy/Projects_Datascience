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
    Read and tokenize newline-separated files.

    Args:
        text_fn (str): Path to the text file.

    Returns:
        iterator: An iterator over tokenized sentences with spaces removed.
    """
    return iter([sent.strip().replace(" ", "")
                 for sent in open(text_fn, encoding="utf8")
                 if sent.strip()])

if __name__ == "__main__":
    # Parse command line arguments using docopt
    args = docopt(__doc__)
    inp_fn = args["--in"]  # Input file name
    out_fn = args["--out"]  # Output file name
    debug = args["--debug"]  # Debug mode flag

    # Set logging level based on debug flag
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Directory containing the base text files
    base_text_dir = "./CADEC_v2/cadec/text/"
    cur_sents = None  # Iterator for current sentences
    cur_sent = None  # Current sentence being processed

    # Open the output file for writing
    with open(out_fn, "w", encoding="utf8") as fout:
        # Process each line in the input file
        for line in tqdm(open(inp_fn, encoding="utf8")):
            line = line.strip()  # Remove leading/trailing whitespace
            data = line.split()  # Split line into tokens

            if len(data) == 1:
                # If the line contains a single token, treat it as a new file reference
                cur_file = data[0]
                cur_sents = read_sents_from_file(os.path.join(base_text_dir, f"{cur_file}.txt"))
                cur_sent = next(cur_sents)  # Get the first sentence from the file
                fout.write(f"{line}\n")  # Write the file reference to output

            elif len(data) == 0:
                # If the line is empty, move to the next sentence
                assert not cur_sent  # Ensure no current sentence is being processed
                try:
                    cur_sent = next(cur_sents)  # Get the next sentence
                except StopIteration:
                    cur_sent = None  # No more sentences in the file
                fout.write("\n")  # Write a blank line to output

            elif len(data) == 5:
                # Skip lines with exactly 5 tokens (specific to your use case)
                continue

            else:
                # Replace the current token with the word from the file
                cur_word = data[0]  # Current word from the input line
                cur_word_len = len(cur_word)  # Length of the current word
                assert cur_word_len > 0  # Ensure the word is non-empty
                assert cur_sent[:cur_word_len].lstrip() == cur_word  # Verify match with the sentence

                # Replace the first token with the word length
                data[0] = str(cur_word_len)
                cur_sent = cur_sent[cur_word_len:]  # Remove the processed word from the sentence

                # Write the modified line to the output file
                fout.write("\t".join(data) + "\n")

    # Log completion message
    logging.info("DONE")