class Registry:
    def __init__(self):
        self._algorithms = {}
    def register_algorithm(self, name, version, constructor):
        key = (name, version)
        self._algorithms[key] = constructor
    def get_algorithm(self, name, version):
        key = (name, version)
        return self._algorithms.get(key)
default_registry = Registry()
