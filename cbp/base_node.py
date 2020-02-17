import uuid
import numpy as np 

from .message import Message

class BaseNode(object):
    def __init__(self,node_coef,potential):
        self.name = str(uuid.uuid4())
        self.node_coef = node_coef
        self.potential = potential
        self.epsilon = 1
        self.coef_ready = False
        self.is_traversed = False
        self.parent = None
        self.node_degree = 0
        self.connections = []
        self.message_inbox = {}
    
    def __str__(self):
        return f"{self.__class__.__name__} {self.node_coef:5.4f}"

    def __repr__(self):
        return self.__str__()

    def format_name(self,name):
        self.name = name
    
    def reset_node_coef(self,coef):
        self.node_coef = coef

    def auto_coef(self,node_map):
        self.node_coef = 1.0 / len(node_map)
        self.register_nodes(node_map)

    # todo, should remove node_map parameter
    def cal_cnp_coef(self):
        raise NotImplementedError(f"{self.__class__.__name__} is an abstract class")

    def check_before_run(self,node_map):
        self.check_connections(self,node_map)

    def check_connections(self,node_map):
        for item in self.connections:
            assert item in node_map, f"{self.name} has a connection {item}, \
                                                    which is not in node_map"

    # keep all message looks urgly. confinient for debug and resource occupied is not so huge
    def store_message(self, message):
        sender_name = message.sender.name
        self.message_inbox[sender_name] = message
        
        self.latest_message = list(self.message_inbox.values())

    def reset(self):
        self.message_inbox.clear()

    def make_message(self,recipient_node):
        raise NotImplementedError(f"{self.__class__.__name__} is an abstract class")

    def send_message(self, recipient_node,is_silent = True):
        val = self.make_message(recipient_node)
        message = Message(self, val)
        recipient_node.store_message(message)
        if not is_silent:
            print(self.name + '->' + recipient_node.name)
            print(message.val)
        
    def sendin_message(self,is_silent = True):
        for connected_node in self.connected_nodes.values():
            connected_node.send_message(self,is_silent)

    def sendout_message(self,is_silent = True):
        for connected_node in self.connected_nodes.values():
            self.send_message(connected_node, is_silent)
        
    def register_connection(self,node_name):
        self.node_degree += 1
        self.connections.append(node_name)

    def register_nodes(self,node_map):
        self.connected_nodes = {}
        for item in self.connections:
            if item in node_map:
                self.connected_nodes[item] = node_map[item]
            else:
                raise IOError(f"connection of {item} of {self.name} \
                                do not appear in the node_map")

    def get_connections(self):
        return self.connections

    def search_node_index(self,node_name):
        return self.connections.index(node_name)

    def search_msg_index(self,message_list,node_name):
        which_index = [i for i,message in enumerate(message_list)\
                            if message.sender.name == node_name]
        if which_index:
            return which_index[0]
        else:
            raise RuntimeError(f"{node_name} do not appear in {self.name} message")

    def __eq__(self, value):
        raise NotImplementedError(f"{self.__class__.__name__} is an abstract class")

    def to_json(self,separators=(',',':'),indent=4):
        raise NotImplementedError(f"{self.__class__.__name__} is an abstract class")

    @classmethod
    def from_json(cls,j):
        raise NotImplementedError(f"{cls.__name__} is an abstract class")

if __name__ == '__main__':
    a = BaseNode()
    print(a)
