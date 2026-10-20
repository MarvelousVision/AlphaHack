
class MLAdapter:
    def __init__(self, unimodel, prepared_data):
        self.unimodel = unimodel
        self.prepared_data = prepared_data

    def __call__(self, plan):
        return self.unimodel(plan, self.prepared_data)
