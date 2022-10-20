import json
from torchfed.utils.object import get_object_size


class RouterMsg(object):
    def __init__(self, from_, to, path, args: tuple):
        self.from_ = from_
        self.to = to
        self.path = path
        self.args = args

    def __str__(self):
        return f"<RouterMsg from={self.from_} to={self.to} path={self.path} args={self.args}>"

    def serialize(self):
        return json.dumps({
            "from": self.from_,
            "to": self.to,
            "path": self.path,
            "args": self.args
        })

    @staticmethod
    def deserialize(data) -> "RouterMsg":
        if isinstance(data, str):
            data = json.loads(data)

        return RouterMsg(
            from_=data["from"],
            to=data["to"],
            path=data["path"],
            args=data["args"])

    @property
    def size(self):
        size = 0
        size += get_object_size(self.from_)
        size += get_object_size(self.to)
        size += get_object_size(self.path)
        size += get_object_size(self.args)
        return size


class RouterMsgResponse(object):
    def __init__(self, from_, to, data):
        self.from_ = from_
        self.to = to
        self.data = data

    def __str__(self):
        return f"<RouterMsgResponse from={self.from_} to={self.to} data={self.data}>"

    @property
    def size(self):
        size = 0
        size += get_object_size(self.from_)
        size += get_object_size(self.to)
        size += get_object_size(self.data)
        return size

    def serialize(self):
        return json.dumps({
            "from": self.from_,
            "to": self.to,
            "data": self.data
        })

    @staticmethod
    def deserialize(data) -> "RouterMsgResponse":
        if isinstance(data, str):
            data = json.loads(data)

        return RouterMsgResponse(
            from_=data["from"],
            to=data["to"],
            data=data["data"])
