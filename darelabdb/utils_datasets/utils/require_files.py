from darelabdb.utils_datasets.dataset_abc import Dataset


def requires_files(method):
    def wrapper(self: Dataset, *args, **kwargs):
        if self.data is None:
            self._init_data()

        return method(self, *args, **kwargs)

    return wrapper
