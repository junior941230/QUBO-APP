def leave_one_file_out_train_sets(file_names):
    """Map each outer test file to a training set that excludes that file."""
    names = list(file_names)
    if len(names) != len(set(names)):
        raise ValueError("File names must be unique for leave-one-file-out")
    return {
        test_file: [name for name in names if name != test_file]
        for test_file in names
    }
