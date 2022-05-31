class RouterMsg(object):
    def __init__(self, from_, to, path, args):
        self.from_ = from_
        self.to = to
        self.path = path
        self.args = args

    def __str__(self):
        return f"<RouterMsg from={self.from_} to={self.to} path={self.path} args={self.args}>"


class RouterMsgResponse(object):
    def __init__(self, from_, to, data):
        self.from_ = from_
        self.to = to
        self.data = data

    def __str__(self):
        return f"<RouterMsgResponse from={self.from_} to={self.to} data={self.data}>"
