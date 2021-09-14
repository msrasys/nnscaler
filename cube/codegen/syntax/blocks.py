

class Block:

    def __init__(self, title):
        if not isinstance(title, str):
            raise TypeError(f"Expected string, but got {type(title)}")
        self.code = [title]

    def __enter__(self):
        return self

    def insert_body(self, code):
        if isinstance(code, list):
            self.code += code
        elif type(code) == str:
            self.code.append(code)
        else:
            raise TypeError

    def __exit__(self, exc_type, exc_value, exc_tb):
        # add indent for function block
        for idx in range(1, len(self.code)):
            self.code[idx] = '\t' + self.code[idx]
        if not exc_tb is None:
            print('Error detected in function block')


class FunctionBlock(Block):

    def __init__(self, func_name, args):
        self.func_name = func_name
        self.param_name = args
        args = ', '.join(args)
        title = f'def {self.func_name}({args}):'
        super().__init__(title)


class ClassBlock(Block):

    def __init__(self, class_name, derived=None):
        if not isinstance(class_name, str):
            raise TypeError("Expected class_name to be str")
        if not isinstance(derived, list) and derived is not None:
            raise TypeError("Expcted derived to be None or list[str]")
        self.class_name = class_name
        if derived:
            derived = ', '.join(derived)
            derived = f'({derived})'
        title = f'class {self.class_name}{derived}'
        super().__init__(self, title)

