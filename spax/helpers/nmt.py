def load_parallel_corpus(src, tgt):
    """
    Load data from the given path of a file in utf-8 format.

    Parameters
    ----------
    src: str
        Path of the source language file to load.
    tgt: str
        Path of the target language file to load.

    Returns
    -------
    src: list
        List of source language sentences.
    tgt: list
        List of target language sentences.
    """
    with open(src, encoding="utf-8") as src_f, open(tgt, encoding="utf-8") as tgt_f:
        src = src_f.readlines()
        tgt = tgt_f.readlines()
        if len(src) != len(tgt):
            raise ValueError("Number of records in source and target files are not equal")
        for i in range(len(src)):
            if not src[i] and tgt[i]:
                src.pop(i)
                tgt.pop(i)
        return src, tgt
    