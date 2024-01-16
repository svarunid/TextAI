import pickle


def sdump(path, lst):
    """
    Dump a list of objects to a file using pickle one by one.

    Parameters
    ----------
    path : str
        Path to the file to write to.
    lst : list
        List of objects to dump.

    Returns
    -------
    None.
    """
    with open(path, "wb") as f:
        for el in lst:
            f.write(pickle.dumps(el))
            f.write(b"\n\n")


def sload(path):
    """
    Streaming itertor for pickled files.

    Parameters
    ----------
    path: str
        Path to the pickle file

    Yields
    ------
    object
        Objects from the pickle file.
    """
    with open(path, "rb") as f:
        cur_lst = []
        for pair in f:
            if pair == b"\n" and cur_lst[-1][-2:] == b".\n":
                cur_lst.append(pair)
                yield pickle.loads(b"".join(cur_lst))
                cur_lst = []
            else:
                cur_lst.append(pair)
