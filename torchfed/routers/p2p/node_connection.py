import base64
import bz2
import json
import lzma
import socket
import threading
import time
import zlib
from struct import unpack, pack
from typing import Union, Any


class NodeConnection(threading.Thread):
    """The class NodeConnection is used by the class Node and represent the TCP/IP socket connection with another node.

       Both inbound (nodes that connect with the server) and outbound (nodes that are connected to) are represented by
       this class. The class contains the client socket and hold the id information of the connecting node.
       Communication is done by this class. When a connecting node sends a message, the message is relayed to the
       main node (that created this NodeConnection in the first place).

       Instantiates a new NodeConnection. Do not forget to start the thread. All TCP/IP communication is handled by this
       connection.
        main_node: The Node class that received a connection.
        sock: The socket that is associated with the client connection.
        id: The id of the connected node (at the other side of the TCP/IP connection).
        host: The host/ip of the main node.
        port: The port of the server of the main node."""

    def __init__(
            self,
            main_node,
            sock: socket.socket,
            id: str,
            host: str,
            port: int):
        super(NodeConnection, self).__init__()

        self.host = host
        self.port = port
        self.main_node = main_node
        self.sock = sock
        self.terminate_flag = threading.Event()

        # The id of the connected node
        self.id = str(id)  # Make sure the ID is a string

        # Start of transmission character for network streaming messages
        self.START_BYTE = 0x01.to_bytes(1, 'big')

        # Length of the packet length in bytes.
        self.PACKET_LENGTH_BYTES = 8

        # End of transmission character for the network streaming messages.
        self.END_BYTE = 0x04.to_bytes(1, 'big')

        # Indication that the message has been compressed
        self.COMPR_CHAR = 0x02.to_bytes(1, 'big')

        # Datastore to store additional information concerning the node.
        self.info = {}

        # Use socket timeout to determine problems with the connection
        self.sock.settimeout(10.0)

        self.main_node.debug_print(
            f"NodeConnection.send: Started with client ({self.id}) '{self.host}:{self.port}'"
        )

    def compress(self, data, compression):
        """Compresses the data given the type. It is used to provide compression to lower the network traffic in case of
           large data chunks. It stores the compression type inside the data, so it can be easily retrieved."""

        self.main_node.debug_print(self.id + ":compress:" + compression)
        self.main_node.debug_print(self.id + ":compress:input: " + str(data))

        compressed = data

        try:
            if compression == 'zlib':
                compressed = base64.b64encode(zlib.compress(data, 6) + b'zlib')

            elif compression == 'bzip2':
                compressed = base64.b64encode(bz2.compress(data) + b'bzip2')

            elif compression == 'lzma':
                compressed = base64.b64encode(lzma.compress(data) + b'lzma')

            else:
                self.main_node.debug_print(
                    self.id + ":compress:Unknown compression")
                return None

        except Exception as e:
            self.main_node.debug_print("compress: exception: " + str(e))

        self.main_node.debug_print(
            self.id +
            ":compress:b64encode:" +
            str(compressed))
        self.main_node.debug_print(self.id +
                                   ":compress:compression:" +
                                   str(int(10000 *
                                           len(compressed) /
                                           len(data)) /
                                       100) +
                                   "%")

        return compressed

    def decompress(self, compressed):
        """Decompresses the data given the type. It is used to provide compression to lower the network traffic in case of
           large data chunks."""
        self.main_node.debug_print(
            self.id +
            ":decompress:input: " +
            str(compressed))
        compressed = base64.b64decode(compressed)
        self.main_node.debug_print(
            self.id +
            ":decompress:b64decode: " +
            str(compressed))

        try:
            if compressed[-4:] == b'zlib':
                compressed = zlib.decompress(compressed[0:len(compressed) - 4])

            elif compressed[-5:] == b'bzip2':
                compressed = bz2.decompress(compressed[0:len(compressed) - 5])

            elif compressed[-4:] == b'lzma':
                compressed = lzma.decompress(compressed[0:len(compressed) - 4])
        except Exception as e:
            print("Exception: " + str(e))

        self.main_node.debug_print(
            self.id +
            ":decompress:result: " +
            str(compressed))

        return compressed

    def send(self, data, encoding_type='utf-8', compression='none'):
        """Send the data to the connected node. The data can be pure text (str), dict object (send as json) and bytes object.
           When sending bytes object, it will be using standard socket communication. A end of transmission character 0x04
           utf-8/ascii will be used to decode the packets ate the other node. When the socket is corrupted the node connection
           is closed. Compression can be enabled by using zlib, bzip2 or lzma. When enabled the data is compressed and send to
           the client. This could reduce the network bandwith when sending large data chunks.
           """
        if isinstance(data, str):
            try:
                self.send_packet(data.encode(encoding_type), compression)
            except Exception as e:  # Fixed issue #19: When sending is corrupted, close the connection
                self.main_node.debug_print(
                    f"nodeconnection send: Error sending data to node: {e}")
                self.stop()  # Stopping node due to failure

        elif isinstance(data, dict):
            try:
                self.send_packet(
                    json.dumps(data).encode(encoding_type),
                    compression)
            except TypeError as type_error:
                self.main_node.debug_print('This dict is invalid')
                self.main_node.debug_print(type_error)

            except Exception as e:  # Fixed issue #19: When sending is corrupted, close the connection
                self.main_node.debug_print(
                    f"nodeconnection send: Error sending data to node: {e}")
                self.stop()  # Stopping node due to failure

        elif isinstance(data, bytes):
            try:
                self.send_packet(data, compression)
            except Exception as e:  # Fixed issue #19: When sending is corrupted, close the connection
                self.main_node.debug_print(
                    "nodeconnection send: Error sending data to node: " + str(e))
                self.stop()  # Stopping node due to failure

        else:
            self.main_node.debug_print(
                'datatype used is not valid please use str, dict (will be send as json) or bytes')

    def send_packet(self, data, compression):
        """Sends the the packet based on the length and encoding type. The packet that will be sent
        consists the following:
        - Start byte
        - Length of packet
        - Data
        - End byte
        """
        if len(data) > pow(2, 64):
            # for now an error is raised
            raise BufferError(
                "Length of the data exceeds the possible length that can be send (8 bytes (2^64))")
            return

        if compression == 'none':
            # packing the package length
            # > indicates the byte order, Q defines the C type

            packet_length = pack('>Q', len(data))
            self.sock.sendall(self.START_BYTE)
            self.sock.sendall(packet_length)
            self.sock.sendall(data)
            self.sock.sendall(self.END_BYTE)
        else:
            compressed_packet = self.compress(data, compression)
            if compressed_packet is not None:
                packet = compressed_packet + self.COMPR_CHAR
                packet_length = pack('>Q', len(packet))
                self.sock.sendall(self.START_BYTE)
                self.sock.sendall(packet_length)
                self.sock.sendall(packet)
                self.sock.sendall(self.END_BYTE)

    def stop(self) -> None:
        """Terminates the connection and the thread is stopped.
        Please make sure you join the thread."""
        self.terminate_flag.set()

    def parse_packet(self, packet) -> Union[str, dict, bytes]:
        """Parse the packet and determines whether it has been send in str, json or byte format. It returns
           the according data."""
        if packet.find(self.COMPR_CHAR) == len(packet) - \
                1:  # Check if packet was compressed
            packet = self.decompress(packet[0:-1])

        try:
            packet_decoded = packet.decode('utf-8')

            try:

                return json.loads(packet_decoded)

            except json.decoder.JSONDecodeError:
                return packet_decoded

        except UnicodeDecodeError:
            return packet

    def run(self):
        """The main loop of the thread to handle the connection with the node. Within the
           main loop the thread waits to receive data from the node. If data is received
           the method node_message will be invoked of the main node to be processed."""
        buffer = b''
        while not self.terminate_flag.is_set():
            chunk = b''

            try:
                chunk = self.sock.recv(4096)

            except socket.timeout:
                self.main_node.debug_print("NodeConnection: timeout")
            except Exception as e:
                self.terminate_flag.set()  # Exception occurred terminating the connection
                self.main_node.debug_print('Unexpected error')
                self.main_node.debug_print(e)

            if chunk != b'':
                buffer += chunk

                start_pos = buffer.find(self.START_BYTE)
                end_pos = buffer.find(self.END_BYTE)

                # if start or end position is not found, the find method
                # returns -1, loop shall be continued. Only if full packets are
                # received the dta shall be extracted.
                if start_pos == -1 or end_pos == -1:
                    time.sleep(0.01)
                    continue

                packet_length = buffer[start_pos + \
                    1:start_pos + self.PACKET_LENGTH_BYTES + 1]
                (unpacked_length,) = unpack('>Q', packet_length)

                if start_pos + self.PACKET_LENGTH_BYTES + unpacked_length + 1 != end_pos:
                    time.sleep(0.01)
                    raise Exception("Error: Incorrect frame construction")

                packet = buffer[start_pos +
                                self.PACKET_LENGTH_BYTES + 1:end_pos]
                buffer = buffer[end_pos + 1:]

                self.main_node.message_count_recv += 1
                self.main_node.node_message(self, self.parse_packet(packet))
                time.sleep(0.01)

        # IDEA: Invoke (event) a method in main_node so the user
        # is able to send a bye message to the node before it is closed?
        self.sock.settimeout(None)
        self.sock.close()
        # Fixed issue #19: Send to main_node when a node is disconnected.
        # We do not know whether it is inbound or outbound.
        self.main_node.node_disconnected(self)
        self.main_node.debug_print("NodeConnection: Stopped")

    def set_info(self, key: str, value: Any) -> Any:
        self.info[key] = value

    def get_info(self, key: str) -> Any:
        return self.info[key]

    def __str__(self) -> str:
        return 'NodeConnection: {}:{} <-> {}:{} ({})'.format(
            self.main_node.host, self.main_node.port, self.host, self.port,
            self.id)

    def __repr__(self) -> str:
        return '<NodeConnection: Node {}:{} <-> Connection {}:{}>'.format(
            self.main_node.host, self.main_node.port, self.host, self.port)
