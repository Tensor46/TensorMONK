import types


class Checks(object):
    def __init__(self, function_name: str):
        assert isinstance(function_name, str)
        self.name = function_name

    def arg_type(self, name, value, exp_type, exp_value):
        if not isinstance(value, exp_type):
            raise TypeError("{}: {} type is {}, expected -- {}".format(
                            self.name, name, type(value).__name__,
                            exp_value))

    def arg_value(self, name, value, exp_value_fnc, exp_value):
        assert isinstance(exp_value_fnc, types.LambdaType), \
            "exp_value_fnc: must be lambda function"
        if not exp_value_fnc(value):
            raise ValueError("{}: {} value is {}, expected -- {}".format(
                             self.name, name, value, exp_value))
