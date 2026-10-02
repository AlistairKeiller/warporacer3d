"""A tiny blocking HTTP/1.0 server on libc sockets, for the local viewer.

One request per connection, served in order; plenty for one browser tab
talking to localhost at 60 Hz, and nothing to install.
"""
from std.ffi import external_call, c_int, c_size_t, c_ssize_t
from std.sys.info import CompilationTarget

comptime AF_INET = 2
comptime SOCK_STREAM = 1
comptime SOL_SOCKET = 0xFFFF if CompilationTarget.is_macos() else 1
comptime SO_REUSEADDR = 0x0004 if CompilationTarget.is_macos() else 2
comptime SIGPIPE = 13
comptime SIG_IGN = 1
comptime LIMIT = 1 << 16


@fieldwise_init
struct Request(Movable):
    var method: String
    var path: String
    var body: String


struct Server:
    var fd: c_int

    def __init__(out self, port: Int) raises:
        if port < 1 or port > 65535:
            raise Error("port must be 1..65535")
        # Writing to a socket the browser already closed must not kill us.
        _ = external_call["signal", Int](c_int(SIGPIPE), Int(SIG_IGN))
        self.fd = external_call["socket", c_int](
            c_int(AF_INET), c_int(SOCK_STREAM), c_int(0)
        )
        if self.fd < 0:
            raise Error("could not create a socket")
        var one = c_int(1)
        _ = external_call["setsockopt", c_int](
            self.fd,
            c_int(SOL_SOCKET),
            c_int(SO_REUSEADDR),
            Pointer(to=one),
            c_int(4),
        )
        # struct sockaddr_in for 127.0.0.1: macOS has a leading length byte,
        # Linux a 16-bit family; both are 16 bytes.
        var address = List[UInt8](length=16, fill=0)
        comptime if CompilationTarget.is_macos():
            address[0] = 16
            address[1] = AF_INET
        else:
            address[0] = AF_INET
        address[2] = UInt8(port >> 8)
        address[3] = UInt8(port & 255)
        address[4] = 127
        address[7] = 1
        if (
            external_call["bind", c_int](
                self.fd, address.unsafe_ptr(), c_int(16)
            )
            != 0
        ):
            raise Error("could not bind 127.0.0.1:" + String(port))
        if external_call["listen", c_int](self.fd, c_int(16)) != 0:
            raise Error("could not listen on port " + String(port))

    def __deinit__(deinit self):
        _ = external_call["close", c_int](self.fd)

    def accept(self) raises -> Connection:
        var client = external_call["accept", c_int](self.fd, Int(0), Int(0))
        if client < 0:
            raise Error("accept failed")
        return Connection(client)


struct Connection(Movable):
    var fd: c_int

    def __init__(out self, fd: c_int):
        self.fd = fd

    def __deinit__(deinit self):
        _ = external_call["close", c_int](self.fd)

    def request(self) raises -> Request:
        """Read one request: the head up to the blank line, then the body."""
        var raw = List[UInt8](capacity=4096)
        var chunk = List[UInt8](length=4096, fill=0)
        var head = -1
        while head < 0:
            var got = external_call["recv", c_ssize_t](
                self.fd, chunk.unsafe_ptr(), c_size_t(len(chunk)), c_int(0)
            )
            if got <= 0:
                raise Error("client closed the connection")
            for i in range(got):
                raw.append(chunk[i])
            if len(raw) > LIMIT:
                raise Error("request too large")
            head = find(raw, "\r\n\r\n")
        var text = String(StringSlice(unsafe_from_utf8=Span(raw))[byte=0:head])
        var lines = text.split("\r\n")
        var words = lines[0].split(" ")
        if len(words) < 2:
            raise Error("malformed request line")
        var length = 0
        for i in range(1, len(lines)):
            var line = lines[i].lower()
            if line.startswith("content-length:"):
                length = Int(line[byte=15:].strip())
        if length < 0 or length > LIMIT:
            raise Error("invalid content length")
        while len(raw) < head + 4 + length:
            var got = external_call["recv", c_ssize_t](
                self.fd, chunk.unsafe_ptr(), c_size_t(len(chunk)), c_int(0)
            )
            if got <= 0:
                raise Error("client closed the connection")
            for i in range(got):
                raw.append(chunk[i])
        var body = String(
            StringSlice(unsafe_from_utf8=Span(raw))[
                byte = head + 4 : head + 4 + length
            ]
        )
        return Request(String(words[0]), String(words[1]), body^)

    def respond(
        self, status: String, content_type: String, body: Span[UInt8, _]
    ) raises:
        var header = String(
            "HTTP/1.0 ",
            status,
            "\r\nContent-Type: ",
            content_type,
            "\r\nContent-Length: ",
            len(body),
            "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        )
        self.send(header.as_bytes())
        self.send(body)

    def send(self, bytes: Span[UInt8, _]) raises:
        var sent = 0
        while sent < len(bytes):
            var wrote = external_call["send", c_ssize_t](
                self.fd,
                bytes.unsafe_ptr().unsafe_offset(sent),
                c_size_t(len(bytes) - sent),
                c_int(0),
            )
            if wrote <= 0:
                raise Error("client closed the connection")
            sent += Int(wrote)


def find(haystack: List[UInt8], needle: StaticString) -> Int:
    var pattern = needle.as_bytes()
    for i in range(len(haystack) - len(pattern) + 1):
        var same = True
        for j in range(len(pattern)):
            if haystack[i + j] != pattern[j]:
                same = False
                break
        if same:
            return i
    return -1
