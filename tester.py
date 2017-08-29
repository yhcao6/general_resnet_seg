from run_parrots import model_and_session


class Tester():
    def __init__(self, model, session, param):
        self.model, self.session = model_and_session(model, session)
        self.session.setup()
        self.flow = self.session.flow('val')
        self.flow.load_param(param)

    def predict(self, inputs, query):
        for k in inputs.keys():
            self.flow.set_input(k, inputs[k])
        self.flow.forward()
        return self.flow.data(query).value().T

