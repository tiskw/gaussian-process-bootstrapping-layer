#!/usr/bin/env python3

"""
Utility Python script to compress the result files.
You don't need to use this script file if you don't mind the file/directory size.
"""

import pathlib


def strip_tsv(path):
    """
    Strip unnecesaty lines from the given result TSV file.
    """
    def keep(line):
        """
        Returns false if the given line should be stripped.
        """
        if   line.startswith("Files already downloaded"): return False
        elif line.startswith("Train: epoch = ")         : return False
        else                                            : return True

    # Create stripped text.
    text = "\n".join([line.rstrip() for line in path.open("rt") if keep(line)])

    # Overwrite to the file.
    path.write_text(text)

    print(path, "done")


def strip_txt(path):
    """
    Strip unnecesaty lines from the given variance TXT file.
    """
    def keep(line):
        """
        Returns false if the given line should be stripped.
        """
        if   line.startswith("Files already downloaded"): return False
        else                                            : return True

    def shorten(line):
        """
        Shorten numerical expression.
        """
        if all(c in ".0123456789\n" for c in line): return "%.2e" % float(line.strip())
        else                                      : return line.rstrip()

    # Create stripped text.
    text = "\n".join([shorten(line) for line in path.open("rt") if keep(line)])

    # Overwrite to the file.
    path.write_text(text)

    print(path, "done")


def main():

    dirpath = pathlib.Path(__file__).parent

    # Process result TSV files.
    for path in dirpath.glob("*.tsv"):
        strip_tsv(path)

    # Process variance TXT files.
    for path in dirpath.glob("*.txt"):
        strip_txt(path)


if __name__ == "__main__": main()


# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
